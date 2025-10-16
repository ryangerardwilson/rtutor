# ~/Apps/rtutor/modules/structs.py
class Lesson:
    def __init__(self, name, content):
        self.name = name  # Lesson name, e.g., "Lesson1"
        self.content = content  # Multiline string for typing practice


class Course:
    def __init__(self, name, lessons):
        self.name = name  # Course name, e.g., "Basic Typing"
        self.lessons = lessons  # List of Lesson objects
