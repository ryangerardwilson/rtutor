# ~/Apps/rtutor/modules/structs.py
class Lesson:
    def __init__(self, name, content):
        self.name = name  # Lesson name, e.g., "Lesson1"
        self.content = content  # Multiline string for typing practice


class Part:
    def __init__(self, name, lessons):
        self.name = name  # Part name, e.g., "Part IA: Chapter 1"
        self.lessons = lessons  # List of Lesson objects


class Course:
    def __init__(self, name, parts):
        self.name = name  # Course name, e.g., "Basic Typing"
        self.parts = parts  # List of Part objects
