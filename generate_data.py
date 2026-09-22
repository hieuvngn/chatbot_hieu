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
BASE_DOCUMENT_KINDS: tuple[str, ...] = ("slides", "textbook")
EXTRA_DOCUMENT_KINDS: tuple[str, ...] = ("exam", "lab_guide", "cheatsheet", "faq")
ALL_DOCUMENT_KINDS: tuple[str, ...] = BASE_DOCUMENT_KINDS + EXTRA_DOCUMENT_KINDS
VIETNAMESE_RATIO_FOR_EXTRA = 0.7
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
# Every prerequisite strictly precedes its course in semester order, so the
# graph is a DAG by construction; assert_no_cycles() enforces the invariant.
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
    (5, "CS316", "Dữ liệu lớn", "Big Data", 3, "Khoa Công nghệ thông tin", ["CS211", "CS223"]),
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
    codes = {c.code: c for c in courses}
    for c in courses:
        for prereq in c.prerequisites:
            assert prereq in codes, f"{c.code} depends on unknown course {prereq}"
            assert (
                codes[prereq].semester < c.semester
            ), f"{c.code} (semester {c.semester}) depends on {prereq} (semester {codes[prereq].semester})"

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


def _section_content(topic_vi: str, topic_en: str, language: str, course: Course, index: int) -> str:
    if language == "vi":
        return (
            f"{topic_vi} là một nội dung quan trọng trong môn {course.name} ({course.code}). "
            f"Trong phần {index}, sinh viên sẽ tìm hiểu khái niệm, cách vận dụng và các ví dụ "
            f"minh họa về {topic_vi}.\n\n"
            f"Đầu tiên, cần phân biệt {topic_vi} với các khái niệm liên quan đã học ở những "
            f"phần trước của học phần. Việc nắm vững định nghĩa chính xác giúp tránh những "
            f"hiểu lầm phổ biến khi làm bài tập vận dụng. Một định nghĩa tốt thường bắt đầu "
            f"từ trực giác: tại sao kỹ thuật này tồn tại, nó giải quyết bài toán gì mà các "
            f"cách tiếp cận đơn giản hơn không giải quyết được, và nó đòi hỏi những giả định "
            f"nào về dữ liệu đầu vào.\n\n"
            f"Tiếp theo, bài giảng trình bày một số thuật toán và kỹ thuật tiêu biểu liên quan "
            f"đến {topic_vi}, kèm theo phân tích độ phức tạp và so sánh giữa các cách tiếp cận. "
            f"Với mỗi thuật toán, chúng ta quan tâm đến ba câu hỏi: độ phức tạp thời gian trong "
            f"trường hợp xấu nhất là bao nhiêu, bộ nhớ phụ trợ cần dùng là bao nhiêu, và việc "
            f"cài đặt có khó hay không. Sự đánh đổi giữa thời gian và bộ nhớ là chủ đề xuyên "
            f"suốt toàn bộ học phần, và {topic_vi} là một ví dụ điển hình của sự đánh đổi này.\n\n"
            f"Cuối phần, có các ví dụ minh họa từng bước và bài tập tự luyện để sinh viên "
            f"kiểm tra mức độ hiểu bài trước khi chuyển sang nội dung kế tiếp. Các bài tập "
            f"được sắp xếp từ dễ đến khó: đầu tiên là những câu hỏi lý thuyết kiểm tra định "
            f"nghĩa, sau đó là những bài tập áp dụng trực tiếp, và cuối cùng là những bài "
            f"toán mở đòi hỏi vận dụng {topic_vi} cùng với kiến thức đã học ở các phần trước. "
            f"Đáp án chi tiết được cung cấp ở cuối tài liệu để sinh viên tự đánh giá.\n\n"
            f"Một lưu ý quan trọng khi học {topic_vi} là không nên học thuộc lòng công thức "
            f"một cách máy móc. Thay vào đó, hãy tập trung vào việc hiểu tại sao công thức "
            f"lại có hình dạng đó, nó xuất phát từ những giả định nào, và trong trường hợp "
            f"nào nó không còn đúng nữa. Khi gặp một bài toán mới, việc đầu tiên cần làm là "
            f"nhận diện xem bài toán thuộc dạng nào, sau đó mới chọn thuật toán phù hợp; "
            f"việc chọn sai thuật toán dù cài đặt đúng vẫn cho kết quả chậm hoặc sai.\n\n"
            f"Bài giảng cũng đề cập đến những lỗi sai thường gặp khi áp dụng {topic_vi} vào "
            f"thực tế, chẳng hạn như nhầm lẫn giữa các trường hợp biên, bỏ sót điều kiện dừng, "
            f"hoặc không kiểm tra tính hợp lệ của dữ liệu đầu vào. Những lỗi này thường xuất "
            f"hiện trong các kỳ thi và bài kiểm tra giữa kỳ, vì vậy sinh viên nên đọc kỹ phần "
            f"này trước khi làm bài tập thực hành."
        )
    return (
        f"{topic_en} is an important topic in {course.name_en} ({course.code}). "
        f"Section {index} covers the concept of {topic_en}, how it is applied, and "
        f"worked examples.\n\n"
        f"First, we distinguish {topic_en} from related concepts covered in earlier "
        f"sections of the module. A precise definition prevents common misunderstandings "
        f"when solving applied exercises. A good definition starts from intuition: why "
        f"this technique exists, what problem it solves that simpler approaches cannot, "
        f"and what assumptions it makes about the input data.\n\n"
        f"Next, the lecture presents representative algorithms and techniques related to "
        f"{topic_en}, with complexity analysis and a comparison of alternative approaches. "
        f"For each algorithm we ask three questions: what is the worst-case time complexity, "
        f"how much auxiliary memory is required, and how hard is the implementation. The "
        f"trade-off between time and memory runs through the whole module, and {topic_en} "
        f"is a typical example of this trade-off.\n\n"
f"The section closes with step-by-step examples and practice problems so students "
        f"can self-assess before moving on. The exercises are ordered from easy to hard: "
        f"first theory questions that check the definition, then direct application "
        f"problems, and finally open-ended tasks that combine {topic_en} with knowledge "
        f"from earlier sections. Detailed solutions are provided at the end of the "
        f"material for self-evaluation.\n\n"
        f"An important note about studying {topic_en}: do not memorize formulas "
        f"mechanically. Instead, focus on understanding why a formula has its shape, "
        f"which assumptions it derives from, and in which cases it no longer holds. "
        f"When facing a new problem, first identify which category it belongs to, then "
        f"choose the appropriate algorithm; choosing the wrong algorithm even with a "
        f"correct implementation still yields slow or wrong results.\n\n"
        f"The lecture also covers common mistakes when applying {topic_en} in practice, "
        f"such as confusing edge cases, omitting termination conditions, or failing to "
        f"validate input data. These mistakes often appear in midterm exams and quizzes, "
        f"so students should read this part carefully before doing the practice exercises.\n\n"
        f"To make the concept concrete, the lecture walks through a complete worked "
        f"example from start to finish: first the problem statement and its constraints, "
        f"then the design of a solution, then the implementation with a short code "
        f"snippet, and finally a trace of the algorithm on a small input showing the "
        f"state at each step. Following the trace by hand on paper is strongly "
        f"recommended, since it builds the mental model needed for harder problems.\n\n"
        f"Related material appears in the reading list: the textbook chapter on "
        f"{topic_en} provides additional exercises, and the slides of the next section "
        f"show how {topic_en} connects to the topics that follow. Students preparing "
        f"for the final exam should treat the practice problems as a self-test, timing "
        f"themselves under exam conditions rather than reading the solutions first."
        )



def _intro_for_kind(language: str, course: Course, kind: str) -> str:
    if kind in {"slides", "textbook"}:
        return _intro_section_legacy(language, course, kind)
    if kind == "exam":
        return _intro_exam(language, course)
    if kind == "lab_guide":
        return _intro_lab_guide(language, course)
    if kind == "cheatsheet":
        return _intro_cheatsheet(language, course)
    if kind == "faq":
        return _intro_faq(language, course)
    return _intro_section_legacy(language, course, kind)


def _summary_for_kind(
    language: str, course: Course, kind: str, topics: list[tuple[str, str]]
) -> str:
    if kind in {"slides", "textbook"}:
        return _summary_section(language, course, topics)
    if kind == "exam":
        return _summary_exam(language, course, topics)
    if kind == "lab_guide":
        return _summary_lab_guide(language, course, topics)
    if kind == "cheatsheet":
        return _summary_cheatsheet(language, course, topics)
    if kind == "faq":
        return _summary_faq(language, course, topics)
    return _summary_section(language, course, topics)


def _section_content_for_kind(
    kind: str,
    topic_vi: str,
    topic_en: str,
    language: str,
    course: Course,
    index: int,
) -> str:
    if kind in {"slides", "textbook"}:
        return _section_content(topic_vi, topic_en, language, course, index)
    if kind == "exam":
        return _section_exam(topic_vi, topic_en, language, course, index)
    if kind == "lab_guide":
        return _section_lab_guide(topic_vi, topic_en, language, course, index)
    if kind == "cheatsheet":
        return _section_cheatsheet(topic_vi, topic_en, language, course, index)
    if kind == "faq":
        return _section_faq(topic_vi, topic_en, language, course, index)
    return _section_content(topic_vi, topic_en, language, course, index)


def _intro_exam(language: str, course: Course) -> str:
    if language == "vi":
        return (
            f"Đề thi mẫu này thuộc môn {course.name} ({course.code}) của "
            f"{course.department}, dành cho sinh viên đã hoàn thành phần lý thuyết "
            f"của học phần. Đề gồm các câu hỏi trắc nghiệm và tự luận kèm đáp án "
            f"chi tiết, bám sát các chủ đề quan trọng nhất.\n\n"
            f"Thời gian làm bài đề xuất là chín mươi phút cho toàn bộ đề; sinh viên "
            f"có thể dành khoảng hai mươi phút để đọc lướt và phân bổ thời gian cho "
            f"từng phần. Cấu trúc đề gồm hai phần chính: phần trắc nghiệm kiểm tra "
            f"khả năng nhớ và phân biệt khái niệm, phần tự luận kiểm tra khả năng "
            f"vận dụng và lập luận. Đáp án chi tiết ở cuối tài liệu giúp sinh viên "
            f"tự đánh giá và ôn lại những điểm còn yếu.\n\n"
            f"Cách sử dụng hiệu quả nhất là làm bài trong điều kiện không có tài "
            f"liệu, sau đó so sánh với đáp án và ghi lại những lỗi sai phổ biến. "
            f"Những lỗi này thường xuất hiện lặp lại trong các kỳ thi, vì vậy sinh "
            f"viên nên phân tích kỹ từng câu sai thay vì chỉ xem đáp án."
        )
    return (
        f"This sample exam belongs to {course.name_en} ({course.code}), offered by "
        f"{course.department}, for students who have completed the theoretical "
        f"part of the module. The paper contains multiple-choice and written "
        f"questions with detailed solutions, focused on the most important topics.\n\n"
        f"Recommended time is ninety minutes for the whole paper; students should "
        f"budget roughly twenty minutes for a quick read-through and time planning. "
        f"The paper has two main parts: multiple-choice items test recall and "
        f"concept discrimination, while the written section tests application and "
        f"reasoning. Detailed solutions at the end of this material let students "
        f"self-assess and revisit the topics they missed.\n\n"
        f"The most effective way to use this paper is to attempt it under timed "
        f"conditions without consulting the material, then compare your answers "
        f"with the solutions and write down the recurring mistakes."
    )


def _intro_lab_guide(language: str, course: Course) -> str:
    if language == "vi":
        return (
            f"Hướng dẫn thực hành này thuộc môn {course.name} ({course.code}) của "
            f"{course.department}, dành cho buổi lab trên lớp. Mỗi bài gồm mục "
            f"tiêu, dữ liệu đầu vào, các bước thực hiện và kết quả mong đợi.\n\n"
            f"Trước mỗi buổi lab, sinh viên cần đọc trước phần mục tiêu và chuẩn "
            f"bị môi trường lập trình theo hướng dẫn ở đầu tài liệu. Trong buổi "
            f"lab, nên làm theo từng bước một và ghi lại kết quả trung gian để "
            f"so sánh với kết quả mong đợi. Sau buổi lab, sinh viên hoàn thành "
            f"phần bài tập mở rộng ở cuối mỗi bài và nộp lại cho giảng viên theo hạn."
        )
    return (
        f"This lab guide belongs to {course.name_en} ({course.code}), offered by "
        f"{course.department}, for in-class lab sessions. Each exercise lists "
        f"objectives, input data, step-by-step instructions, and the expected "
        f"output.\n\n"
        f"Before each session, read the objectives and set up the programming "
        f"environment as described in the front matter. During the session, follow "
        f"the steps in order and record intermediate results so you can compare "
        f"them against the expected output. After the session, complete the "
        f"extension tasks at the end of each exercise and submit them by the "
        f"stated deadline."
    )


def _intro_cheatsheet(language: str, course: Course) -> str:
    if language == "vi":
        return (
            f"Bảng tóm tắt nhanh này dành cho môn {course.name} ({course.code}), "
            f"tổng hợp các công thức, khái niệm và lưu ý quan trọng nhất của "
            f"học phần trên một vài trang. Sinh viên có thể in ra và dùng để ôn "
            f"nhanh trước kỳ thi hoặc khi cần tra cứu khi làm bài tập.\n\n"
            f"Bảng tóm tắt này là điểm khởi đầu, không thay thế cho việc đọc kỹ "
            f"từng phần trong tài liệu chính."
        )
    return (
        f"This cheatsheet summarises the most important formulas, concepts, and "
        f"caveats of {course.name_en} ({course.code}) onto a few pages. Students "
        f"can print it for quick revision before the exam or keep it nearby when "
        f"working through exercises.\n\n"
        f"This cheatsheet is a starting point and does not replace reading the "
        f"main material carefully."
    )


def _intro_faq(language: str, course: Course) -> str:
    if language == "vi":
        return (
            f"Hỏi đáp thường gặp này tổng hợp các câu hỏi mà sinh viên hay hỏi "
            f"về môn {course.name} ({course.code}), gồm cả câu hỏi về nội dung, "
            f"cách học và tổ chức lớp. Trước khi gửi email cho giảng viên hoặc "
            f"trợ giảng, sinh viên nên đọc qua FAQ này; phần lớn câu hỏi về đề "
            f"cương, tiêu chí chấm điểm và thời hạn nộp bài đều đã được trả lời "
            f"ở đây."
        )
    return (
        f"This FAQ collects the questions students most often ask about "
        f"{course.name_en} ({course.code}), covering content, study methods, and "
        f"course organisation. Before emailing the instructor or teaching "
        f"assistant, please skim this FAQ first; most questions about syllabus, "
        f"grading, and deadlines are already covered here."
    )


def _summary_exam(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    names = ", ".join(t for t, _ in topics[:3])
    if language == "vi":
        return (
            f"Tóm tắt đề thi mẫu cho môn {course.name} ({course.code}): các câu "
            f"hỏi xoay quanh {names}. Sinh viên nên luyện làm đề trong điều kiện "
            f"giới hạn thời gian và so sánh kỹ với đáp án ở cuối tài liệu."
        )
    en_names = ", ".join(t for _, t in topics[:3])
    return (
        f"This sample exam summary for {course.name_en} ({course.code}) covers "
        f"{en_names}. Students should attempt the paper under timed conditions "
        f"and compare their answers against the solutions at the end."
    )


def _summary_lab_guide(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    names = ", ".join(t for t, _ in topics[:3])
    if language == "vi":
        return (
            f"Hoàn thành các bài lab về {names} giúp sinh viên làm quen với "
            f"công cụ thực tế của môn {course.name} ({course.code}). Hãy lưu lại "
            f"kết quả từng bước để đối chiếu với kết quả mong đợi khi debug."
        )
    en_names = ", ".join(t for _, t in topics[:3])
    return (
        f"Completing the labs on {en_names} builds hands-on familiarity with "
        f"the tools used in {course.name_en} ({course.code}). Record each "
        f"intermediate result so you can compare it against the expected output "
        f"when debugging."
    )


def _summary_cheatsheet(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    names = ", ".join(t for t, _ in topics[:3])
    if language == "vi":
        return (
            f"Bảng tóm tắt nhanh của môn {course.name} ({course.code}) bao gồm "
            f"các công thức và lưu ý quan trọng nhất về {names}. Dùng để ôn "
            f"nhanh trước kỳ thi."
        )
    en_names = ", ".join(t for _, t in topics[:3])
    return (
        f"This cheatsheet for {course.name_en} ({course.code}) lists the most "
        f"important formulas and caveats about {en_names}. Use it for a quick "
        f"revision before the exam."
    )


def _summary_faq(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    names = ", ".join(t for t, _ in topics[:3])
    if language == "vi":
        return (
            f"FAQ của môn {course.name} ({course.code}) gồm các câu hỏi thường "
            f"gặp xoay quanh {names}. Đọc qua FAQ trước khi gửi email cho giảng "
            f"viên giúp tiết kiệm thời gian cho cả hai phía."
        )
    en_names = ", ".join(t for _, t in topics[:3])
    return (
        f"The FAQ for {course.name_en} ({course.code}) answers the most common "
        f"questions about {en_names}. Reading it before emailing the instructor "
        f"saves time for both sides."
    )


def _section_exam(topic_vi: str, topic_en: str, language: str, course: Course, index: int) -> str:
    if language == "vi":
        return (
            f"Câu hỏi {index} (phần {topic_vi}):\n\n"
            f"a) Định nghĩa {topic_vi} và nêu một ví dụ áp dụng trong môn "
            f"{course.name} ({course.code}).\n"
            f"b) Phân tích độ phức tạp trong trường hợp xấu nhất.\n"
            f"c) So sánh {topic_vi} với một kỹ thuật liên quan đã học.\n\n"
            f"Đáp án gợi ý:\n"
            f"- Phần a: nêu định nghĩa ngắn gọn và một ví dụ cụ thể.\n"
            f"- Phần b: tính toán dựa trên giả định đầu vào.\n"
            f"- Phần c: liệt kê hai điểm giống và hai điểm khác biệt."
        )
    return (
        f"Question {index} (topic {topic_en}):\n\n"
        f"a) Define {topic_en} and give one example applied to "
        f"{course.name_en} ({course.code}).\n"
        f"b) Analyse the worst-case complexity.\n"
        f"c) Compare {topic_en} with a related technique covered earlier.\n\n"
        f"Hint solutions:\n"
        f"- (a) Provide a concise definition and a concrete example.\n"
        f"- (b) Reason under explicit input assumptions.\n"
        f"- (c) List two similarities and two differences."
    )


def _section_lab_guide(topic_vi: str, topic_en: str, language: str, course: Course, index: int) -> str:
    if language == "vi":
        return (
            f"Bài lab {index}: Áp dụng {topic_vi} trong môn {course.name} "
            f"({course.code}).\n\n"
            f"Mục tiêu: cài đặt và chạy thử một chương trình minh họa cho "
            f"{topic_vi}.\n\n"
            f"Các bước thực hiện:\n"
            f"1. Tạo project mới và cài đặt môi trường.\n"
            f"2. Cài đặt hàm xử lý liên quan đến {topic_vi}.\n"
            f"3. Chạy với dữ liệu mẫu và ghi lại kết quả.\n"
            f"4. So sánh với kết quả mong đợi ở cuối bài.\n\n"
            f"Kết quả mong đợi: chương trình in ra đầu ra đúng với dữ liệu mẫu.\n\n"
            f"Bài tập mở rộng: thay đổi dữ liệu đầu vào và quan sát sự thay đổi "
            f"của kết quả; viết một báo cáo ngắn giải thích."
        )
    return (
        f"Lab {index}: Applying {topic_en} in {course.name_en} ({course.code}).\n\n"
        f"Objective: implement and run a small program that illustrates "
        f"{topic_en}.\n\n"
        f"Steps:\n"
        f"1. Create a new project and set up the environment.\n"
        f"2. Implement the function that handles {topic_en}.\n"
        f"3. Run the program on the sample input and record the output.\n"
        f"4. Compare with the expected output at the end of the exercise.\n\n"
        f"Expected output: the program prints the correct result for the sample "
        f"input.\n\n"
        f"Extension: change the input and observe how the output evolves; write a "
        f"short report explaining your observations."
    )


def _section_cheatsheet(topic_vi: str, topic_en: str, language: str, course: Course, index: int) -> str:
    if language == "vi":
        return (
            f"Mục {index}: {topic_vi}.\n\n"
            f"- Định nghĩa ngắn: ghi một dòng về {topic_vi}.\n"
            f"- Công thức chính: liệt kê ký hiệu thường gặp.\n"
            f"- Lưu ý: nhắc lại điều kiện áp dụng và lỗi sai thường gặp.\n"
            f"- Tham chiếu: phần tương ứng trong tài liệu chính của môn "
            f"{course.name} ({course.code})."
        )
    return (
        f"Item {index}: {topic_en}.\n\n"
        f"- One-line definition of {topic_en}.\n"
        f"- Key formulas: list the common notations.\n"
        f"- Caveats: applicability conditions and common mistakes.\n"
        f"- Reference: the corresponding section in the main material of "
        f"{course.name_en} ({course.code})."
    )


def _section_faq(topic_vi: str, topic_en: str, language: str, course: Course, index: int) -> str:
    if language == "vi":
        return (
            f"Câu hỏi FAQ {index}: liên quan đến {topic_vi} trong môn "
            f"{course.name} ({course.code}).\n\n"
            f"Q: {topic_vi} thường gây nhầm lẫn ở điểm nào?\n"
            f"A: Phần lớn sinh viên nhầm lẫn giữa điều kiện áp dụng và trường "
            f"hợp biên. Nên đọc kỹ phần định nghĩa trong tài liệu chính trước "
            f"khi làm bài tập.\n\n"
            f"Q: Tài liệu nào nên đọc thêm về {topic_vi}?\n"
            f"A: Phần tương ứng trong tài liệu chính của học phần và chương sách "
            f"giáo trình được trích dẫn trong phần đọc thêm."
        )
    return (
        f"FAQ {index}: about {topic_en} in {course.name_en} ({course.code}).\n\n"
        f"Q: Where does {topic_en} usually cause confusion?\n"
        f"A: Most students confuse the applicability conditions with the edge "
        f"cases. Read the definition section in the main material carefully "
        f"before attempting the exercises.\n\n"
        f"Q: Which extra material covers {topic_en}?\n"
        f"A: The corresponding section in the main material and the textbook "
        f"chapter listed in the reading list."
    )


def _intro_section_legacy(language: str, course: Course, kind: str) -> str:
    label_vi = "slide" if kind == "slides" else "chương"
    label_en = "slide" if kind == "slides" else "chapter"
    if language == "vi":
        return (
            f"Tài liệu này thuộc môn {course.name} ({course.code}) của "
            f"{course.department}. Đây là {label_vi} giới thiệu tổng quan nội dung "
            f"học phần, gồm các khái niệm cốt lõi và bài tập vận dụng.\n\n"
            f"Học phần có {course.credits} tín chỉ, được giảng dạy ở học kỳ "
            f"{course.semester}. Người học cần nắm vững các môn nền tảng trước khi bắt đầu. "
            f"Cấu trúc của học phần được thiết kế theo lộ trình từ cơ bản đến nâng cao: "
            f"những tuần đầu tập trung vào khái niệm nền tảng, những tuần giữa mở rộng sang "
            f"các kỹ thuật và công cụ thực hành, và những tuần cuối dành cho bài tập tổng hợp "
            f"và đồ án nhỏ. Người học nên đọc trước tài liệu của từng tuần để buổi học trên "
            f"lớp tập trung vào thảo luận và giải bài tập.\n\n"
            f"Mục tiêu của {label_vi} này là giúp sinh viên hình dung toàn bộ mạch kiến thức, "
            f"từ đó biết phần nào là trọng tâm để ưu tiên thời gian ôn tập. Sau khi hoàn thành "
            f"học phần, sinh viên sẽ có thể vận dụng các kiến thức này vào các môn học tiếp "
            f"theo của chương trình đào tạo, đồng thời phát triển kỹ năng tự học và tìm kiếm "
            f"tài liệu tham khảo chuyên ngành.\n\n"
            f"Cách học hiệu quả nhất với tài liệu này là kết hợp đọc trước ở nhà và làm bài "
            f"tập trên lớp. Trước mỗi buổi học, hãy dành khoảng ba mươi phút để đọc phần tương "
            f"ứng và ghi lại những câu hỏi còn thắc mắc; những câu hỏi này sẽ là trọng tâm của "
            f"buổi thảo luận. Sau buổi học, dành mười lăm phút để tóm tắt lại bằng lời của "
            f"mình, vì việc tái diễn đạt kiến thức giúp ghi nhớ lâu hơn nhiều so với việc chỉ "
            f"đọc lại. Phương pháp này áp dụng cho tất cả các {label_vi} của môn học, không "
            f"riêng gì {label_vi} này.\n\n"
            f"Tài liệu cũng liệt kê các mục tiêu học tập dự kiến của học phần: khi kết thúc "
            f"học kỳ, sinh viên có thể giải thích từng khái niệm cốt lõi, vận dụng chúng vào "
            f"các bài toán thực tế, và so sánh một cách phê phán các kỹ thuật hiện có. Các "
            f"mục tiêu này định hướng cả bài giảng lẫn tiêu chí chấm điểm, vì vậy sinh viên "
            f"có thể dùng chúng để tự kiểm tra tiến độ của mình vào cuối mỗi tuần."
        )
    return (
        f"This material belongs to {course.name_en} ({course.code}), offered by "
        f"{course.department}. This {label_en} introduces the module overview, its "
        f"core concepts, and practice exercises.\n\n"
        f"The module carries {course.credits} credits and is taught in semester "
        f"{course.semester}. Students should master the foundation modules first. The "
        f"module is structured from basic to advanced: the first weeks cover fundamental "
        f"concepts, the middle weeks extend into practical techniques and tools, and the "
        f"final weeks are dedicated to integrated assignments and a small project. Students "
        f"are encouraged to read the material of each week in advance so that class time "
        f"can focus on discussion and problem solving.\n\n"
        f"The goal of this {label_en} is to map the knowledge flow of the whole module "
        f"so students know which parts deserve the most review time. Upon completion, "
        f"students will be able to apply these concepts in the next modules of the "
        f"curriculum and develop self-study and technical documentation skills.\n\n"
        f"The most effective way to use this material is to combine reading at home "
        f"with doing exercises in class. Before each session, spend about thirty minutes "
        f"reading the corresponding part and writing down open questions; those questions "
        f"become the focus of the discussion. After the session, spend fifteen minutes "
        f"summarizing in your own words, because restating knowledge helps retention "
        f"far more than rereading. This method applies to every {label_en} of the "
        f"module, not just this one.\n\n"
        f"The material also lists the expected learning outcomes of the module: by "
        f"the end of the term, students should be able to explain each core concept, "
        f"apply it to realistic problems, and critically compare the available "
        f"techniques. These outcomes guide both the lectures and the grading rubric, "
        f"so students can use them to check their own progress at the end of each week.\n\n"
        f"For instructors, this material serves as the reference for every lecture: "
        f"the sequence of sections mirrors the order in which topics are presented "
        f"in class, and the practice problems at the end of each section match the "
        f"difficulty level expected in the midterm and final exams. Office hours and "
        f"consultation sessions build on the material as well, so students who attend "
        f"them should bring the specific questions they noted while reading.\n\n"
        f"Finally, note the assessment structure described in the syllabus: "
        f"continuous assessment accounts for a significant share of the final grade, "
        f"so regular work on the practice problems matters as much as the final exam "
        f"itself. Details of the grading scheme, deadlines, and submission format are "
        f"all available on the module's course page, and this {label_en} should be "
        f"read in combination with that page before the term begins."
    )


def _summary_section(language: str, course: Course, topics: list[tuple[str, str]]) -> str:
    if language == "vi":
        names = ", ".join(t for t, _ in topics[:3])
        return (
            f"Tóm lại, môn {course.name} ({course.code}) giúp sinh viên nắm vững các "
            f"nội dung như {names}.\n\n"
            f"Các phần tiếp theo của học phần sẽ mở rộng những khái niệm này, vì vậy "
            f"người học nên xem lại bài giảng hiện tại trước khi đọc tài liệu tham khảo "
            f"và bài giảng kế tiếp. Điểm mấu chốt cần ghi nhớ là mối liên hệ giữa các "
            f"phần với nhau: mỗi phần đều xây dựng trên kiến thức của phần trước và "
            f"chuẩn bị nền tảng cho phần sau, vì vậy việc ôn tập thường xuyên quan trọng "
            f"hơn việc học dồn vào cuối kỳ.\n\n"
            f"Trước khi chuyển sang tài liệu mới, sinh viên nên tự trả lời ba câu hỏi: "
            f"tôi có thể giải thích các khái niệm chính bằng lời của mình không, tôi có "
            f"thể áp dụng chúng vào một bài tập mới không, và tôi có biết chúng liên hệ "
            f"với các môn học khác như thế nào không. Nếu chưa trả lời được, hãy quay lại "
            f"đọc các phần tương ứng trước khi tiếp tục.\n\n"
            f"Ngoài ba câu hỏi trên, sinh viên cũng nên luyện tập cách trình bày lời giải "
            f"một cách mạch lạc, vì đây là kỹ năng được chấm điểm trực tiếp trong các bài "
            f"kiểm tra. Một lời giải tốt cần nêu rõ giả định, các bước biến đổi chính và "
            f"kết luận cuối cùng; việc viết ra lời giải hoàn chỉnh cũng giúp phát hiện "
            f"những chỗ hiểu chưa đúng mà việc đọc thầm không thể thấy được.\n\n"
            f"Lời khuyên cuối cùng: hãy lập một nhóm học nhỏ gồm hai hoặc ba bạn cùng lớp. "
            f"Giải thích kiến thức cho người khác là một trong những cách nhanh nhất để "
            f"nhận ra mình có thực sự hiểu hay không, và việc chuẩn bị cho một buổi thảo "
            f"luận nhóm buộc bạn phải sắp xếp ghi chú thành một câu chuyện rõ ràng. Hãy "
            f"nhớ rằng phần tóm tắt này là điểm khởi đầu, không phải là sự thay thế cho "
            f"nội dung chi tiết của các phần trong tài liệu.\n\n"
            f"Sinh viên thấy tài liệu dễ có thể chuyển thẳng sang danh mục đọc thêm, gồm "
            f"các chương sách giáo trình và bài báo nghiên cứu về những chủ đề được đề cập. "
            f"Sinh viên thấy khó nên đọc lại các phần tương ứng trước, sau đó thử làm những "
            f"bài tập dễ hơn trước khi xem lời giải. Dù theo hướng nào, điều quan trọng nhất "
            f"là giữ nhịp học ổn định: học phần có tính tích lũy, và chậm một tuần sẽ khiến "
            f"các tuần sau trở nên khó khăn hơn đáng kể."
        )
    names = ", ".join(t for _, t in topics[:3])
    return (
        f"In summary, {course.name_en} ({course.code}) covers topics such as {names}.\n\n"
        f"Later sections of the module build on these concepts, so students should "
        f"review this lecture before moving on to the references and the next lecture. "
        f"The key point to remember is how the sections relate to each other: every "
        f"section builds on the knowledge of the previous one and prepares the ground "
        f"for the next, so regular revision matters more than cramming at the end of "
        f"the term.\n\n"
        f"Before moving to new material, students should answer three questions: can I "
        f"explain the main concepts in my own words, can I apply them to a new exercise, "
        f"and do I know how they relate to other modules. If not, go back to the "
        f"corresponding sections before continuing.\n\n"
        f"Besides these three questions, students should also practice presenting "
        f"solutions coherently, since this skill is graded directly in the exams. A "
        f"good solution states its assumptions, the main transformation steps, and the "
        f"final conclusion; writing out a complete solution also reveals gaps in "
        f"understanding that silent reading cannot expose.\n\n"
        f"A final piece of advice: form a small study group with two or three "
        f"classmates. Explaining a concept to others is one of the fastest ways to "
        f"find out whether you truly understand it, and preparing for a group "
        f"discussion forces you to organize your notes into a clear narrative. "
        f"Remember that the summary of this material is a starting point, not a "
        f"replacement for the detailed content of the sections.\n\n"
        f"Students who found this material easy may move directly to the reading "
        f"list, which contains the textbook chapters and recent research papers on "
        f"the topics covered here. Students who found it difficult should first "
        f"re-read the corresponding sections, then attempt the easier practice "
        f"problems before consulting the solutions. Either way, the important thing "
        f"is to keep a steady pace: the module is cumulative, and falling behind in "
        f"one week makes the following weeks considerably harder.\n\n"
        f"The instructors also maintain a frequently asked questions page where "
        f"common misunderstandings about the module are collected. Before sending an "
        f"email, check that page first; most questions about the syllabus, the "
        f"grading, and the exercise expectations are already answered there. If the "
        f"question is about a specific problem, include your own attempt and the "
        f"point where you got stuck, since that allows the instructor to help much "
        f"more effectively.\n\n"
        f"Students who found this material easy may move directly to the reading "
        f"list, which contains the textbook chapters and recent research papers on "
        f"the topics covered here. Students who found it difficult should first "
        f"re-read the corresponding sections, then attempt the easier practice "
        f"problems before consulting the solutions. Either way, the important thing "
        f"is to keep a steady pace: the module is cumulative, and falling behind in "
        f"one week makes the following weeks considerably harder."
    )


def _document_title(language: str, course: Course, kind: str) -> str:
    vi_map = {
        "slides": "Bài giảng",
        "textbook": "Chương trình học",
        "exam": "Đề thi mẫu",
        "lab_guide": "Hướng dẫn thực hành",
        "cheatsheet": "Bảng tóm tắt nhanh",
        "faq": "Hỏi đáp thường gặp",
    }
    en_map = {
        "slides": "Lecture",
        "textbook": "Study Material",
        "exam": "Sample Exam",
        "lab_guide": "Lab Guide",
        "cheatsheet": "Cheatsheet",
        "faq": "FAQ",
    }
    if language == "vi":
        label = vi_map.get(kind, kind)
        return f"{label}: {course.name} ({course.code})"
    label = en_map.get(kind, kind)
    return f"{label}: {course.name_en} ({course.code})"


def _intro_title(language: str) -> str:
    return "Giới thiệu" if language == "vi" else "Introduction"


def _section_title(language: str, index: int, topic_title: str) -> str:
    if language == "vi":
        return f"Phần {index}: {topic_title}"
    return f"Section {index}: {topic_title}"


def _summary_title(language: str) -> str:
    return "Tóm tắt" if language == "vi" else "Summary"


def _maybe_rewrite(
    content: str,
    kind: str,
    language: str,
    course: Course,
    llm_client: _LLMClient | None,
) -> str:
    if llm_client is None:
        return content
    if kind not in EXTRA_DOCUMENT_KINDS:
        return content
    try:
        rewritten = llm_client.rewrite_section(content, kind, language, course)
    except Exception:
        return content
    return rewritten if rewritten else content


def _document_for_course(
    rng: random.Random,
    course: Course,
    kind: str,
    language: str,
    doc_id: str,
    llm_client: _LLMClient | None = None,
) -> Document:
    topics = list(TOPICS[course.code])
    rng.shuffle(topics)
    num_topics = rng.randint(2, min(4, len(topics)))
    selected = topics[:num_topics]

    title = _document_title(language, course, kind)
    intro_text = _maybe_rewrite(
        _intro_for_kind(language, course, kind), kind, language, course, llm_client
    )
    chapters = [
        Chapter(
            id=f"{doc_id}-c1",
            title=_intro_title(language),
            content=intro_text,
        )
    ]
    for i, (tvi, ten) in enumerate(selected, start=2):
        topic_title = tvi if language == "vi" else ten
        chapters.append(
            Chapter(
                id=f"{doc_id}-c{i}",
                title=_section_title(language, i - 1, topic_title),
                content=_section_content_for_kind(
                    kind, tvi, ten, language, course, i - 1
                ),
            )
        )
    summary_text = _maybe_rewrite(
        _summary_for_kind(language, course, kind, selected),
        kind,
        language,
        course,
        llm_client,
    )
    chapters.append(
        Chapter(
            id=f"{doc_id}-c{len(chapters) + 1}",
            title=_summary_title(language),
            content=summary_text,
        )
    )
    return Document(
        id=doc_id,
        course_code=course.code,
        title=title,
        kind=kind,
        language=language,
        chapters=chapters,
    )


def _build_documents(
    rng: random.Random,
    courses: list[Course],
    count: int = NUM_DOCUMENTS,
    kinds: tuple[str, ...] = BASE_DOCUMENT_KINDS,
    start_index: int = 1,
) -> list[Document]:
    """Build ``count`` documents whose ids run from ``DOC-{start_index:03d}`` upward.

    Default arguments preserve the original deterministic behaviour (40 docs,
    half ``slides``/half ``textbook``); passing ``count`` and/or ``kinds`` is
    used by the extra-documents layer but never by the legacy tests.
    """
    pool = list(courses)
    rng.shuffle(pool)
    chosen = pool[:count]
    if len(chosen) < count:
        repeats = (count + len(pool) - 1) // len(pool)
        chosen = (chosen * repeats)[:count]

    languages = ["vi"] * (count // 2) + ["en"] * (count // 2)
    languages = languages[:count]
    kinds_seq = (list(kinds) * ((count // len(kinds)) + 1))[:count]
    rng.shuffle(languages)
    rng.shuffle(kinds_seq)

    return [
        _document_for_course(
            rng, course, kind, language, f"DOC-{start_index + i:03d}"
        )
        for i, (course, kind, language) in enumerate(
            zip(chosen, kinds_seq, languages)
        )
    ]


class _LLMClient:
    """Minimal seam the generator uses to ask an LLM to rewrite a section.

    Concrete implementations live outside the generator so the CLI can wire
    OpenRouter (or a no-op for tests). Failures must return ``None`` so the
    generator falls back to the deterministic template.
    """

    def rewrite_section(
        self, content: str, kind: str, language: str, course: "Course"
    ) -> str | None:
        return None


def _build_extra_documents(
    rng: random.Random,
    courses: list[Course],
    count: int,
    llm_client: _LLMClient | None = None,
) -> list[Document]:
    """Extra documents in the four new ``kind``s, biased toward Vietnamese.

    ``count`` is added on top of the 40 base documents; IDs run from
    ``DOC-041`` upward. When ``llm_client`` is provided, intro/summary
    sections for the new kinds may be rewritten by the LLM; any failure
    silently falls back to the deterministic template.
    """
    if count <= 0:
        return []
    pool = list(courses)
    rng.shuffle(pool)
    chosen = (pool * ((count // len(pool)) + 1))[:count]

    vi_n = round(count * VIETNAMESE_RATIO_FOR_EXTRA)
    languages: list[str] = ["vi"] * vi_n + ["en"] * (count - vi_n)
    rng.shuffle(languages)

    kinds_seq = (
        list(EXTRA_DOCUMENT_KINDS)
        * ((count // len(EXTRA_DOCUMENT_KINDS)) + 1)
    )[:count]
    rng.shuffle(kinds_seq)

    documents: list[Document] = []
    for i, (course, kind, language) in enumerate(
        zip(chosen, kinds_seq, languages)
    ):
        doc_id = f"DOC-{NUM_DOCUMENTS + i + 1:03d}"
        documents.append(
            _document_for_course(
                rng, course, kind, language, doc_id, llm_client=llm_client
            )
        )
    return documents


def generate(
    seed: int = SEED,
    num_documents: int = NUM_DOCUMENTS,
    extra_documents: int = 0,
    llm_client: _LLMClient | None = None,
) -> Dataset:
    rng = random.Random(seed)
    courses = _build_courses(rng)
    documents = _build_documents(rng, courses, count=num_documents)
    documents.extend(
        _build_extra_documents(rng, courses, extra_documents, llm_client=llm_client)
    )
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


class _OpenRouterClient(_LLMClient):
    """Optional LLM client wired to OpenRouter; only used when ``--use-llm`` is set.

    The OpenAI-compatible endpoint and model follow the same defaults as the
    rest of the project. Missing key disables the client (returns ``None``)
    so the generator falls back to the deterministic template.
    """

    @classmethod
    def from_env(cls) -> _OpenRouterClient | None:
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL

        return cls(
            api_key=api_key,
            model=os.environ.get("RAG_LLM_MODEL", DEFAULT_LLM_MODEL),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        )

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def rewrite_section(
        self, content: str, kind: str, language: str, course: "Course"
    ) -> str | None:
        prompt = (
            f"You are helping generate study material for an IT course chatbot. "
            f"Rewrite the following {kind} section in {language}, keeping the same "
            f"course ({course.code}). The output must mention the course by code "
            f"and reference topics from the course. Output only the rewritten text."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
            )
        except Exception:
            return None
        text = (response.choices[0].message.content or "").strip()
        return text or None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic CourseMate demo dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility (default: %(default)s).")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output data directory (default: %(default)s).")
    parser.add_argument(
        "--extra-documents",
        type=int,
        default=0,
        help=(
            "Number of extra documents to add on top of the 40 base documents. "
            "Extras use the new kinds (exam, lab_guide, cheatsheet, faq) and are "
            "biased toward Vietnamese. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help=(
            "Rewrite intro/summary sections of extra documents via OpenRouter. "
            "Requires OPENROUTER_API_KEY; failures silently fall back to the "
            "deterministic template."
        ),
    )
    args = parser.parse_args(argv)

    llm_client: _LLMClient | None = None
    if args.use_llm:
        llm_client = _OpenRouterClient.from_env()

    dataset = generate(
        seed=args.seed,
        extra_documents=args.extra_documents,
        llm_client=llm_client,
    )
    write(dataset, args.out)
    print(
        f"Generated {len(dataset.courses)} courses and {len(dataset.documents)} "
        f"documents in {args.out}"
    )



if __name__ == "__main__":
    main()