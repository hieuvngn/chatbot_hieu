from __future__ import annotations

import json
from pathlib import Path

import generate_data as gd

SEED = 42

TOPIC_KEYWORDS_VI = {
    "mảng", "ngăn xếp", "hàng đợi", "cây", "bảng băm", "đồ thị", "đệ quy",
    "thuật toán", "đối tượng", "hàm", "biến", "con trỏ", "bộ nhớ", "ngôn ngữ",
    "cơ sở dữ liệu", "mạng máy tính", "bảo mật", "mã hóa", "hệ điều hành",
    "tiến trình", "lập trình", "phần mềm", "kiểm thử", "học máy", "mạng nơ-ron",
    "xác suất", "thống kê", "phân tích dữ liệu", "đạo hàm", "tích phân", "ma trận",
    "không gian vectơ", "logic", "tập hợp", "toán rời rạc", "đồ họa", "điện toán",
    "đám mây", "an toàn thông tin", "thiết kế", "kiến trúc", "xử lý ảnh",
    "ngôn ngữ tự nhiên", "dữ liệu lớn", "project", "dự án", "hàm băm", "khóa",
    "mã hóa bất đối xứng", "chuỗi", "xâu ký tự", "sắp xếp", "tìm kiếm",
    "giải tích", "hàm số", "định thức", "đại số tuyến tính", "chuỗi số",
    "đạo hàm riêng", "phương trình vi phân", "phân phối", "kiểm định",
    "giao thức", "tầng mạng", "tiếng Anh chuyên ngành", "thuật ngữ", "bài báo khoa học",
    "chuẩn hóa", "giao tác", "vòng đời", "mô hình", "kiến trúc phần mềm",
    "tác tử", "biểu diễn tri thức", "tấn công", "chữ ký số", "máy ảo", "luồng",
    "đa luồng", "định tuyến", "chuyển mạch", "chất lượng dịch vụ", "mạng không dây",
    "dữ liệu huấn luyện", "dự đoán", "đánh giá", "tách từ", "phân tích cú pháp",
    "ngữ nghĩa", "ứng dụng di động", "giao diện người dùng", "cảm biến",
    "phép biến đổi", "ánh sáng", "bề mặt", "độ phủ", "kiểm thử tự động",
    "lỗi phần mềm", "phân tán", "luồng dữ liệu", "lan truyền ngược", "hàm kích hoạt",
    "trực quan hóa", "xu hướng", "dự báo", "ảo hóa", "hạ tầng", "dịch vụ",
    "kỹ năng học tập", "quản lý thời gian", "ghi chú", "làm việc nhóm",
    "vật lý", "hoạt hình", "xâm nhập", "phòng thủ", "giám sát", "báo cáo",
    "hệ thống", "mô hình dự đoán", "thử nghiệm", "đồng bộ", "hiệu năng",
    "người dùng", "trải nghiệm", "kinh doanh", "đầu tư", "quyền sở hữu",
    "trách nhiệm", "doanh nghiệp", "hồ sơ", "kinh nghiệm", "nghiên cứu",
    "luận văn", "bảo vệ", "công nghệ mới", "kế hoạch", "ngân sách", "rủi ro",
    "đặc tả", "nghiệp vụ", "mô hình hóa", "tiền mã hóa", "hợp đồng thông minh",
    "sổ cái", "hệ thống nhúng", "vi điều khiển", "văn hóa", "cộng đồng",
    "cấu trúc điều khiển", "lớp", "kế thừa", "đa hình", "tối ưu hóa",
    "quy hoạch tuyến tính", "hàm mục tiêu", "ràng buộc", "danh sách", "từ điển",
    "vòng lặp", "thư viện", "độ phức tạp", "chia để trị", "cấu trúc dữ liệu",
    "chuyên đề", "khởi nghiệp", "thực tập", "đồ án", "quản trị dự án",
    "phân tích yêu cầu", "đạo đức", "luật", "kinh doanh", "bài tập", "khái niệm",
    "ví dụ", "khóa", "tài liệu", "học phần", "sinh viên",
}

TOPIC_KEYWORDS_EN = {
    "array", "stack", "queue", "tree", "hash table", "graph", "recursion",
    "algorithm", "object", "function", "variable", "pointer", "memory",
    "language", "database", "network", "security", "encryption", "operating system",
    "process", "programming", "software", "testing", "machine learning",
    "neural network", "probability", "statistics", "data analysis", "derivative",
    "integral", "matrix", "vector space", "logic", "set theory", "discrete math",
    "graphics", "computing", "cloud", "information security", "design",
    "architecture", "image processing", "natural language", "big data",
    "project", "hash function", "key", "asymmetric", "string", "sorting",
    "search", "digital logic", "compiler", "automata",
    "calculus", "determinant", "linear algebra", "series", "partial derivative",
    "differential equation", "distribution", "hypothesis testing", "protocol",
    "network layer", "terminology", "research paper", "normalization",
    "transaction", "lifecycle", "model", "software architecture", "agent",
    "knowledge representation", "attack", "digital signature", "virtual machine",
    "thread", "multithreading", "routing", "switching", "quality of service",
    "wireless network", "training data", "prediction", "evaluation",
    "tokenization", "parsing", "semantics", "mobile application", "user interface",
    "sensor", "transformation", "lighting", "surface", "coverage",
    "automated testing", "software defect", "distributed", "data stream",
    "backpropagation", "activation function", "visualization", "trend",
    "forecasting", "virtualization", "infrastructure", "service", "study skills",
    "time management", "note-taking", "teamwork", "physics", "animation",
    "intrusion", "defense", "monitoring", "report", "system", "predictive model",
    "experiment", "synchronization", "performance", "user", "experience",
    "business", "investment", "intellectual property", "responsibility",
    "company", "portfolio", "experience", "research", "dissertation", "defense",
    "emerging technology", "plan", "budget", "risk", "specification",
    "modeling", "cryptocurrency", "smart contract", "ledger", "embedded system",
    "iot", "microcontroller", "culture", "community", "control flow", "class",
    "inheritance", "polymorphism", "optimization", "linear programming",
    "objective function", "constraint", "list", "dictionary", "loop", "library",
    "complexity", "divide and conquer", "data structure", "capstone",
    "entrepreneurship", "internship", "thesis", "project management",
    "requirements engineering", "ethics", "law", "concept", "example", "key",
    "material", "module", "student", "cpu", "sql", "html", "css", "javascript",
    "python", "java", "android", "hadoop", "blockchain", "turing machine",
    "grammar", "formal language",
}


def dataset() -> gd.Dataset:
    return gd.generate(seed=SEED)


def test_generates_50_courses() -> None:
    ds = dataset()
    assert len(ds.courses) == 50


def test_every_course_has_all_required_fields() -> None:
    ds = dataset()
    for c in ds.courses:
        assert c.code
        assert c.name
        assert c.name_en
        assert isinstance(c.credits, int) and 1 <= c.credits <= 6
        assert isinstance(c.prerequisites, list)
        assert isinstance(c.semester, int) and 1 <= c.semester <= 8
        assert c.department
        assert c.instructor
        assert c.description


def test_course_codes_are_unique() -> None:
    ds = dataset()
    codes = [c.code for c in ds.courses]
    assert len(codes) == len(set(codes))


def test_prerequisite_codes_refer_to_known_courses() -> None:
    ds = dataset()
    codes = {c.code for c in ds.courses}
    for c in ds.courses:
        for prereq in c.prerequisites:
            assert prereq in codes, f"{c.code} depends on unknown course {prereq}"


def test_prerequisite_graph_is_acyclic() -> None:
    ds = dataset()
    graph = {c.code: set(c.prerequisites) for c in ds.courses}

    indegree = {node: len(deps) for node, deps in graph.items()}
    queue = [node for node, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for other, deps in graph.items():
            if node in deps:
                indegree[other] -= 1
                if indegree[other] == 0:
                    queue.append(other)

    assert visited == len(graph), "prerequisite graph contains a cycle"


def test_generate_asserts_acyclicity_without_raising() -> None:
    dataset()


def test_generates_40_documents() -> None:
    ds = dataset()
    assert len(ds.documents) == 40


def test_documents_reference_known_courses() -> None:
    ds = dataset()
    codes = {c.code for c in ds.courses}
    for d in ds.documents:
        assert d.course_code in codes


def test_documents_are_mixed_vietnamese_and_english() -> None:
    ds = dataset()
    langs = {d.language for d in ds.documents}
    assert langs == {"vi", "en"}


def test_documents_have_chapters_with_content() -> None:
    ds = dataset()
    for d in ds.documents:
        assert d.kind in {"slides", "textbook"}
        assert d.title
        assert len(d.chapters) >= 2
        for ch in d.chapters:
            assert ch.title
            assert ch.content


def test_document_content_references_its_course() -> None:
    ds = dataset()
    courses = {c.code: c for c in ds.courses}
    for d in ds.documents:
        course = courses[d.course_code]
        text = " ".join(ch.content for ch in d.chapters).lower()
        assert (
            course.code.lower() in text
            or course.name.lower() in text
            or course.name_en.lower() in text
        ), f"document {d.id} does not reference its course {course.code}"


def test_document_content_references_topics() -> None:
    ds = dataset()
    for d in ds.documents:
        text = " ".join(ch.content for ch in d.chapters).lower()
        keywords = TOPIC_KEYWORDS_VI if d.language == "vi" else TOPIC_KEYWORDS_EN
        hits = [kw for kw in keywords if kw.lower() in text]
        assert hits, f"document {d.id} ({d.language}) references no known topic"


def test_generation_is_deterministic_for_same_seed() -> None:
    first = gd.generate(seed=7)
    second = gd.generate(seed=7)
    assert first == second


def test_generation_varies_across_seeds() -> None:
    a = gd.generate(seed=7)
    b = gd.generate(seed=8)
    assert a != b


def test_write_creates_json_files(tmp_path: Path) -> None:
    ds = dataset()
    gd.write(ds, tmp_path)
    courses_file = tmp_path / "courses.json"
    documents_file = tmp_path / "documents.json"
    assert courses_file.exists()
    assert documents_file.exists()

    courses = json.loads(courses_file.read_text())
    documents = json.loads(documents_file.read_text())
    assert len(courses) == 50
    assert len(documents) == 40


def test_writing_is_deterministic(tmp_path: Path) -> None:
    gd.write(gd.generate(seed=3), tmp_path)
    files_1 = {
        name: (tmp_path / name).read_bytes()
        for name in ("courses.json", "documents.json")
    }

    other = tmp_path / "other"
    gd.write(gd.generate(seed=3), other)
    files_2 = {
        name: (other / name).read_bytes()
        for name in ("courses.json", "documents.json")
    }

    assert files_1 == files_2


def test_cli_writes_data_dir() -> None:
    from subprocess import run

    result = run(
        ["python", "-m", "generate_data", "--seed", str(SEED)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr