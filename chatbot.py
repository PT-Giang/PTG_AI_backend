import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC FLASHCARD & QUIZ
# ==========================================

class Flashcard(BaseModel):
    front: str = Field(description="Mặt trước: Thuật ngữ, câu hỏi hoặc khái niệm chính")
    back: str = Field(description="Mặt sau: Giải thích ngắn gọn, định nghĩa hoặc câu trả lời")
    source_page: str = Field(description="Trích dẫn nguồn: Tên file và số trang")

class FlashcardSet(BaseModel):
    topic: str = Field(description="Chủ đề chính của bộ flashcard")
    cards: List[Flashcard] = Field(description="Danh sách các flashcard")

class QuizQuestion(BaseModel):
    question: str = Field(description="Câu hỏi kiểm tra")
    options: List[str] = Field(description="4 phương án lựa chọn A, B, C, D")
    correct_answer: str = Field(description="Đáp án đúng (chính xác văn bản phương án)")
    explanation: str = Field(description="Giải thích chi tiết lý do chọn đáp án này dựa trên tài liệu")

class QuizSet(BaseModel):
    title: str = Field(description="Tiêu đề bài kiểm tra")
    questions: List[QuizQuestion] = Field(description="Danh sách câu hỏi trắc nghiệm")


# ==========================================
# 2. KHỞI TẠO RAG BASE
# ==========================================

def format_docs(docs):
    if not docs:
        return "Không tìm thấy thông tin phù hợp trong tài liệu."
    return "\n\n".join([f"(Nguồn: {doc.metadata.get('source', 'Unknown')}, Trang: {doc.metadata.get('page', 'N/A')})\n{doc.page_content}" for doc in docs])

def setup_rag_system(force_recreate: bool = False):
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "papers_collection")

    # 1. Kiểm tra kết nối tới Qdrant Server (Docker)
    try:
        client = QdrantClient(url=qdrant_url, timeout=5)
        client.get_collections()
    except Exception as e:
        raise ConnectionError(
            f"\n[LỖI KẾT NỐI] Không thể kết nối tới Qdrant server tại: {qdrant_url}\n"
            f"Vui lòng đảm bảo container Qdrant đang chạy qua Docker:\n"
            f"  -> Khởi động: docker compose up -d\n"
            f"Chi tiết: {e}"
        )

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    # 2. Kiểm tra xem collection đã tồn tại và có vectors chưa
    has_collection = client.collection_exists(collection_name=collection_name)
    has_vectors = False
    if has_collection:
        collection_info = client.get_collection(collection_name=collection_name)
        has_vectors = (collection_info.points_count or 0) > 0

    if not has_vectors or force_recreate:
        print(f"[*] Đang đọc tài liệu và lưu vector embeddings vào Qdrant (Collection: '{collection_name}')...")
        # Load & Vectorize 
        loaders = DirectoryLoader(path="./papers", glob="**/*.docx", loader_cls=UnstructuredFileLoader, loader_kwargs={
            "strategy": "fast",
            "languages": ["vie", "eng"]  
        }, show_progress=True, use_multithreading=True)
        docs = loaders.load()
        if not docs:
            print("[CẢNH BÁO] Không tìm thấy tài liệu docx nào trong thư mục ./papers!")

        text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount=0.85)
        splits = text_splitter.split_documents(docs)

        vectorstores = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name,
            force_recreate=force_recreate,
        )
        print(f"[THÀNH CÔNG] Đã vector hóa và lưu {len(splits)} chunks vào Qdrant!")
    else:
        print(f"[*] Đã tìm thấy collection '{collection_name}' trên Qdrant ({collection_info.points_count} vectors). Tải kết nối nhanh...")
        vectorstores = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    retriever = vectorstores.as_retriever(search_kwargs={"k": 6})
    return retriever

# ==========================================
# 3. TẠO CÁC CHAIN GIÁO VIÊN AI
# ==========================================

def get_teacher_chains(retriever):
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        temperature=0.2
    )

    # --- CHAIN 1: TẠO FLASHCARD ---
    flashcard_parser = JsonOutputParser(pydantic_object=FlashcardSet)
    flashcard_template = (
        "Bạn là một giáo viên tận tâm. Hãy phân tích ngữ cảnh được cung cấp từ tài liệu và trích xuất ra các khái niệm, "
        "định nghĩa hoặc thuật ngữ quan trọng nhất để tạo thành một bộ Flashcard giúp học sinh ghi nhớ bài học.\n\n"
        "Định dạng đầu ra BẮT BUỘC tuân thủ JSON theo hướng dẫn:\n{format_instructions}\n\n"
        "Ngữ cảnh tài liệu:\n{context}\n\n"
        "Chủ đề yêu cầu: {question}"
    )
    flashcard_prompt = ChatPromptTemplate.from_template(
        template=flashcard_template,
        partial_variables={"format_instructions": flashcard_parser.get_format_instructions()}
    )
    flashcard_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | flashcard_prompt
        | llm
        | flashcard_parser
    )

    # --- CHAIN 2: TẠO BÀI KIỂM TRA (QUIZ) ---
    quiz_parser = JsonOutputParser(pydantic_object=QuizSet)
    quiz_template = (
        "Bạn là một giáo viên chuyên nghiệp. Dựa vào ngữ cảnh tài liệu được cung cấp, hãy biên soạn một bài kiểm tra trắc nghiệm "
        "để đánh giá mức độ hiểu bài của học sinh. Các câu hỏi phải bám sát nội dung và có lời giải thích rõ ràng.\n\n"
        "Định dạng đầu ra BẮT BUỘC tuân thủ JSON theo hướng dẫn:\n{format_instructions}\n\n"
        "Ngữ cảnh tài liệu:\n{context}\n\n"
        "Yêu cầu bài test: {question}"
    )
    quiz_prompt = ChatPromptTemplate.from_template(
        template=quiz_template,
        partial_variables={"format_instructions": quiz_parser.get_format_instructions()}
    )
    quiz_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | quiz_prompt
        | llm
        | quiz_parser
    )

    return flashcard_chain, quiz_chain

# ==========================================
# 4. GIAO DIỆN CHẠY CHƯƠNG TRÌNH
# ==========================================

if __name__ == "__main__":
    print("Đang khởi tạo Trợ lý Giáo viên AI...")
    retriever = setup_rag_system()
    flashcard_chain, quiz_chain = get_teacher_chains(retriever)
    
    while True:
        print("\n=== TRỢ LÝ GIÁO VIÊN AI ===")
        print("1. Tạo bộ Flashcard học tập")
        print("2. Tạo Bài kiểm tra trắc nghiệm")
        print("3. Nạp lại dữ liệu (Re-index) vào Qdrant")
        print("4. Thoát")
        
        choice = input("Chọn chức năng (1-4): ").strip()
        
        if choice == "4":
            break
            
        if choice == "3":
            print("\nĐang tiến hành re-index lại toàn bộ tài liệu vào Qdrant...")
            retriever = setup_rag_system(force_recreate=True)
            flashcard_chain, quiz_chain = get_teacher_chains(retriever)
            print("Đã cập nhật vector database thành công!")
            continue
        
        topic = input("Nhập chủ đề/bài học bạn muốn tạo (Ví dụ: 'Giao thức MQTT' hoặc 'Chương 1'): ")
        
        if choice == "1":
            print("\nAI đang soạn Flashcard, vui lòng đợi...")
            result = flashcard_chain.invoke(f"Tạo bộ flashcard về chủ đề: {topic}")
            
            print(f"\n BỘ FLASHCARD: {result.get('topic', '')}")
            print("="*50)
            for i, card in enumerate(result.get('cards', []), 1):
                print(f"Card {i}:")
                print(f"  [Mặt trước]: {card['front']}")
                print(f"  [Mặt sau]  : {card['back']}")
                print(f"  [Trích dẫn]: {card['source_page']}\n")

        elif choice == "2":
            print("\nAI đang biên soạn Bài kiểm tra, vui lòng đợi...")
            result = quiz_chain.invoke(f"Tạo bài kiểm tra 3-5 câu hỏi về chủ đề: {topic}")
            
            print(f"\n BÀI KIỂM TRA: {result.get('title', '')}")
            print("="*50)
            for i, q in enumerate(result.get('questions', []), 1):
                print(f"Câu {i}: {q['question']}")
                for opt in q['options']:
                    print(f"   {opt}")
                print(f"   >> Đáp án đúng: {q['correct_answer']}")
                print(f"   >> Giải thích  : {q['explanation']}\n")