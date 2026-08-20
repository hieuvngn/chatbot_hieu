from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

SEED = 42
NUM_COURSES = 50
NUM_DOCUMENTS = 40
DATA_DIR = Path(__file__).parent / "data"

INSTRUCTORS = [
    "Nguyễn Văn Anh",
    "Trần Thị Bình",
    "Lê Văn Cường",
    "Phạm Thu Hà",
    "Hoàng Minh Đức",
    "Vũ Thị Hương",
    "Đặng Quốc Huy",
    "Bùi Thanh Hằng",
    "Đỗ Hồng Nhung",
    "Ngô Đức Thắng",
    "Trương Quang Vinh",
    "Lý Gia Bảo",
    "Võ Thu Trang",
    "Hà Anh Tuấn",
    "Dương Thúy An",
    "Phan Công Khánh",
    "Nguyễn Hải Long",
    "Lâm Nhật Minh",
    "Cao Thiên Phúc",
    "Đinh Ngọc Sơn",
]


@dataclass
class Course:
    code: str
    name: str
    name_en: str
    credits: int
    prerequisites: list[str]
    semester: int
    department: str
    instructor: str
    description: str


@dataclass
class Chapter:
    id: str
    title: str
    content: str


@dataclass
class Document:
    id: str
    course_code: str
    title: str
    kind: str
    language: str
    chapters: list[Chapter] = field(default_factory=list)


@dataclass
class Dataset:
    courses: list[Course]
    documents: list[Document]


# (vi, en) topic phrases per course. Phrases are drawn from the same
# vocabulary the knowledge-Q&A tests assert on.
TOPICS: dict[str, list[tuple[str, str]]] = {
    "CS101": [("lập trình", "programming"), ("biến", "variable"), ("hàm", "function"),
              ("cấu trúc điều khiển", "control flow")],
    "CS102": [("toán rời rạc", "discrete math"), ("tập hợp", "set theory"),
              ("logic", "logic"), ("đồ thị", "graph")],
    "MATH101": [("giải tích", "calculus"), ("đạo hàm", "derivative"),
                ("tích phân", "integral"), ("hàm số", "function")],
    "MATH102": [("ma trận", "matrix"), ("không gian vectơ", "vector space"),
                ("đại số tuyến tính", "linear algebra"), ("định thức", "determinant")],
    "CS103": [("kiến trúc", "architecture"), ("bộ nhớ", "memory"),
              ("CPU", "cpu"), ("bus hệ thống", "system bus")],
    "CS104": [("kỹ năng học tập", "study skills"), ("quản lý thời gian", "time management"),
              ("ghi chú", "note-taking"), ("làm việc nhóm", "teamwork")],
    "CS111": [("đối tượng", "object"), ("lớp", "class"), ("kế thừa", "inheritance"),
              ("đa hình", "polymorphism")],
    "CS112": [("cấu trúc dữ liệu", "data structure"), ("mảng", "array"),
              ("ngăn xếp", "stack"), ("hàng đợi", "queue"), ("cây", "tree"),
              ("bảng băm", "hash table")],
    "MATH111": [("chuỗi số", "series"), ("đạo hàm riêng", "partial derivative"),
                ("phương trình vi phân", "differential equation")],
    "STAT101": [("xác suất", "probability"), ("thống kê", "statistics"),
                ("phân phối", "distribution"), ("kiểm định", "hypothesis testing")],
    "CS113": [("mạng máy tính", "network"), ("giao thức", "protocol"),
              ("IP", "ip"), ("TCP", "tcp"), ("tầng mạng", "network layer")],
    "GE101": [("tiếng Anh chuyên ngành", "technical english"), ("thuật ngữ", "terminology"),
              ("bài báo khoa học", "research paper")],
    "CS211": [("cơ sở dữ liệu", "database"), ("SQL", "sql"), ("khóa", "key"),
              ("chuẩn hóa", "normalization"), ("giao tác", "transaction")],
    "CS212": [("lập trình Web", "web programming"), ("HTML", "html"), ("CSS", "css"),
              ("JavaScript", "javascript"), ("giao diện", "interface")],
    "CS213": [("hệ điều hành", "operating system"), ("tiến trình", "process"),
              ("bộ nhớ ảo", "virtual memory"), ("lập lịch", "scheduling")],
    "CS214": [("ngôn ngữ hình thức", "formal language"), ("ô-tô-mát", "automata"),
              ("văn phạm", "grammar"), ("máy Turing", "turing machine")],
    "MATH211": [("tối ưu hóa", "optimization"), ("quy hoạch tuyến tính", "linear programming"),
                ("hàm mục tiêu", "objective function"), ("ràng buộc", "constraint")],
    "CS215": [("Python", "python"), ("danh sách", "list"), ("từ điển", "dictionary"),
              ("vòng lặp", "loop"), ("thư viện", "library")],
    "CS221": [("thuật toán", "algorithm"), ("độ phức tạp", "complexity"),
              ("sắp xếp", "sorting"), ("tìm kiếm", "search"), ("chia để trị", "divide and conquer")],
    "CS222": [("phần mềm", "software"), ("vòng đời", "lifecycle"), ("mô hình", "model"),
              ("kiến trúc phần mềm", "software architecture")],
    "CS223": [("trí tuệ nhân tạo", "artificial intelligence"), ("tìm kiếm", "search"),
              ("tác tử", "agent"), ("biểu diễn tri thức", "knowledge representation")],
    "CS224": [("an toàn thông tin", "information security"), ("mã hóa", "encryption"),
              ("bảo mật", "security"), ("tấn công", "attack"), ("chữ ký số", "digital signature")],
    "CS225": [("Java", "java"), ("máy ảo", "virtual machine"), ("luồng", "thread"),
              ("đa luồng", "multithreading")],
    "CS226": [("định tuyến", "routing"), ("chuyển mạch", "switching"),
              ("chất lượng dịch vụ", "quality of service"), ("mạng không dây", "wireless network")],
    "CS311": [("học máy", "machine learning"), ("dữ liệu huấn luyện", "training data"),
              ("mô hình", "model"), ("dự đoán", "prediction"), ("đánh giá", "evaluation")],
    "CS312": [("ngôn ngữ tự nhiên", "natural language"), ("tách từ", "tokenization"),
              ("phân tích cú pháp", "parsing"), ("ngữ nghĩa", "semantics")],
    "CS313": [("ứng dụng di động", "mobile application"), ("Android", "android"),
              ("giao diện người dùng", "user interface"), ("cảm biến", "sensor")],
    "CS314": [("đồ họa", "graphics"), ("phép biến đổi", "transformation"),
              ("ánh sáng", "lighting"), ("bề mặt", "surface")],
    "CS315": [("kiểm thử", "testing"), ("độ phủ", "coverage"), ("kiểm thử tự động", "automated testing"),
              ("lỗi phần mềm", "software defect")],
    "CS316": [("dữ liệu lớn", "big data"), ("Hadoop", "hadoop"), ("phân tán", "distributed"),
              ("luồng dữ liệu", "data stream")],
    "CS321": [("học sâu", "deep learning"), ("mạng nơ-ron", "neural network"),
              ("lan truyền ngược", "backpropagation"), ("hàm kích hoạt", "activation function")],
    "CS322": [("phân tích dữ liệu", "data analysis"), ("trực quan hóa", "visualization"),
              ("xu hướng", "trend"), ("dự báo", "forecasting")],
    "CS323": [("đám mây", "cloud"), ("ảo hóa", "virtualization"),
              ("hạ tầng", "infrastructure"), ("dịch vụ", "service")],
    "CS324": [("lập trình hệ thống", "systems programming"), ("con trỏ", "pointer"),
              ("bộ nhớ", "memory"), ("ngôn ngữ C", "c language")],
    "CS325": [("phát triển game", "game development"), ("engine", "engine"),
              ("vật lý", "physics"), ("hoạt hình", "animation")],
    "CS326": [("an ninh mạng", "cybersecurity"), ("xâm nhập", "intrusion"),
              ("phòng thủ", "defense"), ("giám sát", "monitoring")],
    "CS411": [("chuyên đề", "capstone"), ("dự án", "project"), ("báo cáo", "report"),
              ("hệ thống", "system")],
    "CS412": [("khoa học dữ liệu", "data science"), ("học máy", "machine learning"),
              ("mô hình dự đoán", "predictive model"), ("thử nghiệm", "experiment")],
    "CS413": [("lập trình song song", "parallel programming"), ("luồng", "thread"),
              ("đồng bộ", "synchronization"), ("hiệu năng", "performance")],
    "CS414": [("tương tác người-máy", "human-computer interaction"), ("người dùng", "user"),
              ("trải nghiệm", "experience"), ("đánh giá", "evaluation")],
    "CS415": [("khởi nghiệp", "entrepreneurship"), ("kinh doanh", "business"),
              ("mô hình", "model"), ("đầu tư", "investment")],
    "CS416": [("luật", "law"), ("đạo đức", "ethics"), ("quyền sở hữu", "intellectual property"),
              ("trách nhiệm", "responsibility")],
    "CS417": [("thực tập", "internship"), ("doanh nghiệp", "company"),
              ("hồ sơ", "portfolio"), ("kinh nghiệm", "experience")],
    "CS421": [("đồ án", "thesis"), ("nghiên cứu", "research"), ("luận văn", "dissertation"),
              ("bảo vệ", "defense")],
    "CS422": [("chuyên đề tiên tiến", "advanced topics"), ("nghiên cứu", "research"),
              ("xu hướng", "trend"), ("công nghệ mới", "emerging technology")],
    "CS423": [("quản trị dự án", "project management"), ("kế hoạch", "plan"),
              ("ngân sách", "budget"), ("rủi ro", "risk")],
    "CS424": [("phân tích yêu cầu", "requirements engineering"), ("đặc tả", "specification"),
              ("nghiệp vụ", "business"), ("mô hình hóa", "modeling")],
    "CS425": [("blockchain", "blockchain"), ("tiền mã hóa", "cryptocurrency"),
              ("hợp đồng thông minh", "smart contract"), ("sổ cái", "ledger")],
    "CS426": [("hệ thống nhúng", "embedded system"), ("IoT", "iot"),
              ("vi điều khiển", "microcontroller"), ("cảm biến", "sensor")],
    "CS427": [("đạo đức nghề nghiệp", "professional ethics"), ("trách nhiệm", "responsibility"),
              ("văn hóa", "culture"), ("cộng đồng", "community")],
}


# Semester, code, vi name, en name, credits, department, prerequisites.
# Prerequisites always reference strictly-earlier semesters, so the graph is
# a DAG by construction.
COURSE_TABLE: list[tuple[int, str, str, str, int, str, list[str]]] = [
    (1, "CS101", "Nhập môn lập trình", "Introduction to Programming", 3, "Khoa Công nghệ thông tin", []),
    (1, "CS102", "Toán rời rạc", "Discrete Mathematics", 3, "Khoa Công nghệ thông tin", []),
    (1, "MATH101", "Giải tích 1", "Calculus I", 3, "Khoa Toán - Thống kê", []),
    (1, "MATH102", "Đại số tuyến tính", "Linear Algebra", 3, "Khoa Toán - Thống kê", []),
    (1, "CS103", "Kiến trúc máy tính", "Computer Architecture", 3, "Khoa Công nghệ thông tin", []),
    (1, "CS104", "Kỹ năng học đại học", "University Study Skills", 2, "Bộ môn Đại cương", []),
    (2, "CS111", "Lập trình hướng đối tượng", "Object-Oriented Programming", 3, "Khoa Công nghệ thông tin", ["CS101"]),
    (2, "CS112", "Cấu trúc dữ liệu và giải thuật", "Data Structures and Algorithms", 4, "Khoa Công nghệ thông tin", ["CS101"]),
    (2, "MATH111", "Giải tích 2", "Calculus II", 3, "Khoa Toán - Thống kê", ["MATH101"]),
    (2, "STAT101", "Xác suất thống kê", "Probability and Statistics", 3, "Khoa Toán - Thống kê", ["MATH101"]),
    (2, "CS113", "Mạng máy tính", "Computer Networks", 3, "Khoa Công nghệ thông tin", ["CS101"]),
    (2, "GE101", "Tiếng Anh chuyên ngành", "Technical English", 2, "Bộ môn Đại cương", []),
    (3, "CS211", "Cơ sở dữ liệu", "Database Systems", 3, "Khoa Công nghệ thông tin", ["CS112"]),
    (3, "CS212", "Lập trình Web", "Web Programming", 3, "Khoa Công nghệ thông tin", ["CS111"]),
    (3, "CS213", "Hệ điều hành", "Operating Systems", 3, "Khoa Công nghệ thông tin", ["CS103"]),
    (3, "CS214", "Ngôn ngữ hình thức", "Formal Languages and Automata", 3, "Khoa Công nghệ thông tin", ["CS102", "CS112"]),
    (3, "MATH211", "Tối ưu hóa", "Optimization", 3, "Khoa Toán - Thống kê", ["MATH111", "STAT101"]),
    (3, "CS215", "Lập trình Python", "Python Programming", 3, "Khoa Công nghệ thông tin", ["CS111"]),
    (4, "CS221", "Phân tích và thiết kế giải thuật", "Algorithm Analysis and Design", 3, "Khoa Công nghệ thông tin", ["CS112", "CS214"]),
    (4, "CS222", "Kỹ thuật phần mềm", "Software Engineering", 3, "Khoa Công nghệ thông tin", ["CS211", "CS212"]),
    (4, "CS223", "Trí tuệ nhân tạo", "Artificial Intelligence", 3, "Khoa Công nghệ thông tin", ["CS112", "STAT101"]),
    (4, "CS224", "An toàn thông tin", "Information Security", 3, "Khoa Công nghệ thông tin", ["CS113", "CS213"]),
    (4, "CS225", "Lập trình Java", "Java Programming", 3, "Khoa Công nghệ thông tin", ["CS111"]),
    (4, "CS226", "Mạng nâng cao", "Advanced Networking", 3, "Khoa Công nghệ thông tin", ["CS113"]),
    (5, "CS311", "Học máy", "Machine Learning", 4, "Khoa Công nghệ thông tin", ["CS223", "MATH211"]),
    (5, "CS312", "Xử lý ngôn ngữ tự nhiên", "Natural Language Processing", 3, "Khoa Công nghệ thông tin", ["CS223"]),
    (5, "CS313", "Phát triển ứng dụng di động", "Mobile Application Development", 3, "Khoa Công nghệ thông tin", ["CS212"]),
    (5, "CS314", "Đồ họa máy tính", "Computer Graphics", 3, "Khoa Công nghệ thông tin", ["MATH111", "CS112"]),
    (5, "CS315", "Kiểm thử phần mềm", "Software Testing", 3, "Khoa Công nghệ thông tin", ["CS222"]),
    (5, "CS316", "Dữ liệu lớn", "Big Data", 3, "Khoa Công nghệ thông tin", ["CS211", "CS311"]),
    (6, "CS321", "Học sâu", "Deep Learning", 4, "Khoa Công nghệ thông tin", ["CS311"]),
    (6, "CS322", "Phân tích dữ liệu", "Data Analytics", 3, "Khoa Công nghệ thông tin", ["STAT101", "CS316"]),
    (6, "CS323", "Điện toán đám mây", "Cloud Computing", 3, "Khoa Công nghệ thông tin", ["CS213", "CS316"]),
    (6, "CS324", "Lập trình hệ thống", "Systems Programming", 3, "Khoa Công nghệ thông tin", ["CS213", "CS225"]),
    (6, "CS325", "Phát triển game", "Game Development", 3, "Khoa Công nghệ thông tin", ["CS314"]),
    (6, "CS326", "An ninh mạng", "Cybersecurity", 3, "Khoa Công nghệ thông tin", ["CS224"]),
    (7, "CS411", "Chuyên đề trí tuệ nhân tạo", "AI Capstone", 3, "Khoa Công nghệ thông tin", ["CS321"]),
    (7, "CS412", "Khoa học dữ liệu", "Data Science", 3, "Khoa Công nghệ thông tin", ["CS316", "CS322"]),
    (7, "CS413", "Lập trình song song", "Parallel Programming", 3, "Khoa Công nghệ thông tin", ["CS213"]),
    (7, "CS414", "Tương tác người-máy", "Human-Computer Interaction", 3, "Khoa Công nghệ thông tin", ["CS222"]),
    (7, "CS415", "Khởi nghiệp CNTT", "IT Entrepreneurship", 2, "Khoa Công nghệ thông tin", ["CS222"]),
    (7, "CS416", "Luật CNTT và đạo đức", "IT Law and Ethics", 2, "Khoa Công nghệ thông tin", []),
    (7, "CS417", "Thực tập", "Internship", 3, "Khoa Công nghệ thông tin", ["CS222"]),
    (8, "CS421", "Đồ án tốt nghiệp", "Graduation Thesis", 6, "Khoa Công nghệ thông tin", ["CS411"]),
    (8, "CS422", "Chuyên đề tiên tiến", "Advanced Topics in Computing", 3, "Khoa Công nghệ thông tin", ["CS321"]),
    (8, "CS423", "Quản trị dự án CNTT", "IT Project Management", 3, "Khoa Công nghệ thông tin", ["CS222"]),
    (8, "CS424", "Phân tích yêu cầu", "Requirements Engineering", 3, "Khoa Công nghệ thông tin", ["CS222"]),
    (8, "CS425", "Blockchain và tiền mã hóa", "Blockchain and Cryptocurrency", 3, "Khoa Công nghệ thông tin", ["CS224"]),
    (8, "CS426", "IoT và hệ thống nhúng", "IoT and Embedded Systems", 3, "Khoa Công nghệ thông tin", ["CS213", "CS324"]),
    (8, "CS427", "Đạo đức nghề nghiệp", "Professional Ethics", 2, "Khoa Công nghệ thông tin", []),
]


def _build_courses(rng: random.Random) -> list[Course]:
    courses: list[Course] = []
    for semester, code, name, name_en, credits, department, prereqs in COURSE_TABLE:
        topics_vi = [t[0] for t in TOPICS[code]]
        year = (semester + 1) // 2
        description = (
            f"Môn học {name} ({code}) thuộc {department}, gồm {credits} tín chỉ, "
            f"giảng dạy ở học kỳ {semester} dành cho sinh viên năm {year}. "
            f"Nội dung chính: {', '.join(topics_vi)}."
        )
        courses.append(
            Course(
                code=code,
                name=name,
                name_en=name_en,
                credits=credits,
                prerequisites=list(prereqs),
                semester=semester,
                department=department,
                instructor=rng.choice(INSTRUCTORS),
                description=description,
            )
        )
    return courses


def assert_no_cycles(courses: Sequence[Course]) -> None:
    """Topological sort; raises AssertionError if the prerequisite graph cycles."""
    codes = {c.code for c in courses}
    for c in courses:
        for prereq in c.prerequisites:
            assert prereq in codes, f"{c.code} depends on unknown course {prereq}"

    indegree = {c.code: 0 for c in courses}
    dependents: dict[str, list[str]] = {c.code: [] for c in courses}
    for c in courses:
        for prereq in c.prerequisites:
            indegree[c.code] += 1
            dependents[prereq].append(c.code)

    queue = [code for code, deg in indegree.items() if deg == 0]
    visited: list[str] = []
    while queue:
        current = queue.pop()
        visited.append(current)
        for next_code in dependents[current]:
            indegree[next_code] -= 1
            if indegree[next_code] == 0:
                queue.append(next_code)

    assert len(visited) == len(courses), "prerequisite graph contains a cycle"


def _sentence(topic_vi: str, topic_en: str, language: str, course: Course) -> str:
    if language == "vi":
        return (
            f"{topic_vi} là một nội dung quan trọng trong môn {course.name} ({course.code}). "
            f"Trong phần này, sinh viên sẽ tìm hiểu khái niệm, cách vận dụng và các ví dụ "
            f"minh họa về {topic_vi}."
        )
    return (
        f"{topic_en} is an important topic in {course.name_en} ({course.code}). "
        f"This section covers the concept of {topic_en}, how it is applied, and "
        f"worked examples."
    )


def _intro_sentence(language: str, course: Course, kind: str) -> str:
    label_vi = "slide" if kind == "slides" else "chương"
    label_en = "slide" if kind == "slides" else "chapter"
    if language == "vi":
        return (
            f"Tài liệu này thuộc môn {course.name} ({course.code}) của "
            f"{course.department}. Đây là {label_vi} giới thiệu tổng quan nội dung "
            f"học phần, gồm các khái niệm cốt lõi và bài tập vận dụng."
        )
    return (
        f"This material belongs to {course.name_en} ({course.code}), offered by "
        f"{course.department}. This {label_en} introduces the module overview, its "
        f"core concepts, and practice exercises."
    )


def _summary_sentence(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    if language == "vi":
        names = ", ".join(t for t, _ in topics[:3])
        return (
            f"Tóm lại, môn {course.name} ({course.code}) giúp sinh viên nắm vững các "
            f"nội dung như {names}. Hãy xem thêm bài giảng tiếp theo và tài liệu tham "
            f"khảo của học phần."
        )
    names = ", ".join(t for _, t in topics[:3])
    return (
        f"In summary, {course.name_en} ({course.code}) covers topics such as {names}. "
        f"See the next lecture and the module references for details."
    )


def _document_for_course(rng: random.Random, course: Course, kind: str, language: str, doc_id: str) -> Document:
    topics = list(TOPICS[course.code])
    rng.shuffle(topics)
    num_topics = rng.randint(2, min(4, len(topics)))
    selected = topics[:num_topics]

    if language == "vi":
        type_label = "Bài giảng" if kind == "slides" else "Chương trình học"
        title = f"{type_label}: {course.name} ({course.code})"
    else:
        type_label = "Lecture" if kind == "slides" else "Study Material"
        title = f"{type_label}: {course.name_en} ({course.code})"

    chapters = [Chapter(id=f"{doc_id}-c1", title="Giới thiệu" if language == "vi" else "Introduction",
                        content=_intro_sentence(language, course, kind))]
    for i, (tvi, ten) in enumerate(selected, start=2):
        topic_title = tvi if language == "vi" else ten
        chapters.append(
            Chapter(id=f"{doc_id}-c{i}",
                    title=f"Phần {i - 1}: {topic_title}" if language == "vi" else f"Section {i - 1}: {topic_title}",
                    content=_sentence(tvi, ten, language, course))
        )
    chapters.append(
        Chapter(id=f"{doc_id}-c{len(chapters) + 1}",
                title="Tóm tắt" if language == "vi" else "Summary",
                content=_summary_sentence(language, course, selected))
    )
    return Document(id=doc_id, course_code=course.code, title=title, kind=kind, language=language,
                    chapters=chapters)


def _build_documents(rng: random.Random, courses: list[Course]) -> list[Document]:
    pool = list(courses)
    rng.shuffle(pool)
    chosen = pool[:NUM_DOCUMENTS]

    languages = ["vi"] * (NUM_DOCUMENTS // 2) + ["en"] * (NUM_DOCUMENTS // 2)
    kinds = (["slides", "textbook"] * (NUM_DOCUMENTS // 2))[:NUM_DOCUMENTS]
    rng.shuffle(languages)
    rng.shuffle(kinds)

    return [
        _document_for_course(rng, course, kind, language, f"DOC-{i + 1:03d}")
        for i, (course, kind, language) in enumerate(zip(chosen, kinds, languages))
    ]


def generate(seed: int = SEED) -> Dataset:
    rng = random.Random(seed)
    courses = _build_courses(rng)
    documents = _build_documents(rng, courses)
    dataset = Dataset(courses=courses, documents=documents)
    assert_no_cycles(dataset.courses)
    return dataset


def _to_json(dataset: Dataset) -> dict[str, object]:
    return {
        "courses": [
            {
                "code": c.code,
                "name": c.name,
                "name_en": c.name_en,
                "credits": c.credits,
                "prerequisites": c.prerequisites,
                "semester": c.semester,
                "department": c.department,
                "instructor": c.instructor,
                "description": c.description,
            }
            for c in dataset.courses
        ],
        "documents": [
            {
                "id": d.id,
                "course_code": d.course_code,
                "title": d.title,
                "kind": d.kind,
                "language": d.language,
                "chapters": [{"id": ch.id, "title": ch.title, "content": ch.content} for ch in d.chapters],
            }
            for d in dataset.documents
        ],
    }


def write(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _to_json(dataset)
    (out_dir / "courses.json").write_text(json.dumps(data["courses"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "documents.json").write_text(json.dumps(data["documents"], ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic CourseMate demo dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility (default: %(default)s).")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output data directory (default: %(default)s).")
    args = parser.parse_args(argv)

    dataset = generate(seed=args.seed)
    write(dataset, args.out)
    print(f"Generated {len(dataset.courses)} courses and {len(dataset.documents)} documents in {args.out}")


if __name__ == "__main__":
    main()