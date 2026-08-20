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


def _intro_section(language: str, course: Course, kind: str) -> str:
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
                        content=_intro_section(language, course, kind))]
    for i, (tvi, ten) in enumerate(selected, start=2):
        topic_title = tvi if language == "vi" else ten
        chapters.append(
            Chapter(id=f"{doc_id}-c{i}",
                    title=f"Phần {i - 1}: {topic_title}" if language == "vi" else f"Section {i - 1}: {topic_title}",
                    content=_section_content(tvi, ten, language, course, i - 1))
        )
    chapters.append(
        Chapter(id=f"{doc_id}-c{len(chapters) + 1}",
                title="Tóm tắt" if language == "vi" else "Summary",
                content=_summary_section(language, course, selected))
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