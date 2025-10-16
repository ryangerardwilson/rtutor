# ~/Apps/rtutor/modules/course_parser.py
import os
from modules.structs import Course, Lesson


class CourseParser:
    def __init__(self, courses_dir):
        self.courses_dir = os.path.abspath(courses_dir)
        print(f"CourseParser initialized with directory: {self.courses_dir}")

    def parse_courses(self):
        """Parse all .md files in the courses_dir into a list of Course objects."""
        courses = []
        if not os.path.isdir(self.courses_dir):
            raise FileNotFoundError(f"Directory {self.courses_dir} does not exist")

        print(f"Scanning directory: {self.courses_dir}")
        for filename in os.listdir(self.courses_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.courses_dir, filename)
                print(f"Processing file: {filepath}")
                course = self._parse_md_file(filepath)
                if course:
                    print(f"Successfully parsed course: {course.name}")
                    courses.append(course)
                else:
                    print(f"Failed to parse course from: {filepath}")
        print(f"Total courses found: {len(courses)}")
        return courses

    def _parse_md_file(self, filepath):
        """Parse a single .md file into a Course object."""
        course_name = None
        lessons = []
        current_lesson_name = None
        lesson_content = []
        in_code_block = False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    print(f"Reading line: '{line}'")

                    # Handle course name (# heading)
                    if line.startswith("# "):
                        if course_name:
                            print(f"Error: Multiple course names in {filepath}")
                            return None
                        course_name = line[2:].strip()
                        print(f"Found course name: {course_name}")
                        continue

                    # Handle lesson name (## heading)
                    if line.startswith("## "):
                        if current_lesson_name and lesson_content:
                            lessons.append(
                                Lesson(current_lesson_name, "\n".join(lesson_content))
                            )
                            print(
                                f"Saved lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                            )
                            lesson_content = []
                        current_lesson_name = line[3:].strip()
                        in_code_block = False
                        print(f"Found lesson name: {current_lesson_name}")
                        continue

                    # Detect code block (indented with 4 spaces or tab)
                    if current_lesson_name and (
                        line.startswith("    ") or line.startswith("\t")
                    ):
                        in_code_block = True
                        # Strip leading 4 spaces or tab for content
                        content_line = line[4:] if line.startswith("    ") else line[1:]
                        lesson_content.append(content_line)
                        print(f"Added to lesson content: '{content_line}'")
                        continue

                    # Preserve blank lines within code block
                    if in_code_block and not line.strip():
                        lesson_content.append("")
                        print(f"Added blank line to lesson content")
                        continue

                    # End code block on non-indented, non-empty line
                    if (
                        in_code_block
                        and line.strip()
                        and not (line.startswith("    ") or line.startswith("\t"))
                    ):
                        in_code_block = False
                        print("Code block end (non-indented line)")

                # Save the last lesson if it exists
                if current_lesson_name and lesson_content:
                    lessons.append(
                        Lesson(current_lesson_name, "\n".join(lesson_content))
                    )
                    print(
                        f"Saved final lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                    )

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

        if course_name and lessons:
            print(f"Created course: {course_name} with {len(lessons)} lessons")
            return Course(course_name, lessons)
        print(f"No valid course or lessons in {filepath}")
        return None
