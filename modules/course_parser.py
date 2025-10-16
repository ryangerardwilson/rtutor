# ~/Apps/rtutor/modules/course_parser.py
import os
from modules.structs import Course, Part, Lesson


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
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

        # Check if the file has hierarchical structure (### for lessons)
        has_hierarchy = any(line.lstrip().startswith("### ") for line in lines)

        if not has_hierarchy:
            # Flat structure: # course, ## lesson, indented content
            # Wrap lessons in a single "Main" part for consistency
            lessons = []
            current_lesson_name = None
            lesson_content = []
            in_code_block = False

            for line in lines:
                line = line.rstrip("\n")
                print(f"Reading line: '{line}'")

                if line.startswith("# "):
                    if course_name:
                        print(f"Error: Multiple course names in {filepath}")
                        return None
                    course_name = line[2:].strip()
                    print(f"Found course name: {course_name}")
                    continue

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

                if current_lesson_name and (
                    line.startswith("    ") or line.startswith("\t")
                ):
                    in_code_block = True
                    content_line = line[4:] if line.startswith("    ") else line[1:]
                    lesson_content.append(content_line)
                    print(f"Added to lesson content: '{content_line}'")
                    continue

                if in_code_block and not line.strip():
                    lesson_content.append("")
                    print(f"Added blank line to lesson content")
                    continue

                if (
                    in_code_block
                    and line.strip()
                    and not (line.startswith("    ") or line.startswith("\t"))
                ):
                    in_code_block = False
                    print("Code block end (non-indented line)")

            # Save the last lesson
            if current_lesson_name and lesson_content:
                lessons.append(Lesson(current_lesson_name, "\n".join(lesson_content)))
                print(
                    f"Saved final lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                )

            if course_name and lessons:
                parts = [Part("Main", lessons)]
                print(
                    f"Created flat course: {course_name} with 1 part and {len(lessons)} lessons"
                )
                return Course(course_name, parts)
            print(f"No valid course or lessons in {filepath}")
            return None

        else:
            # Hierarchical: # course, ## part, ### lesson, indented content
            parts = []
            current_part = None
            current_part_name = None
            current_lesson_name = None
            lesson_content = []
            in_code_block = False

            for line in lines:
                line = line.rstrip("\n")
                print(f"Reading line: '{line}'")

                if line.startswith("# "):
                    if course_name:
                        print(f"Error: Multiple course names in {filepath}")
                        return None
                    course_name = line[2:].strip()
                    print(f"Found course name: {course_name}")
                    continue

                if line.startswith("## "):
                    if current_lesson_name and lesson_content:
                        current_part.lessons.append(
                            Lesson(current_lesson_name, "\n".join(lesson_content))
                        )
                        print(
                            f"Saved lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                        )
                        lesson_content = []
                    if current_part:
                        parts.append(current_part)
                        print(f"Saved part: {current_part_name}")
                    current_part_name = line[3:].strip()
                    current_part = Part(current_part_name, [])
                    current_lesson_name = None
                    in_code_block = False
                    print(f"Found part name: {current_part_name}")
                    continue

                if line.startswith("### "):
                    if current_lesson_name and lesson_content:
                        current_part.lessons.append(
                            Lesson(current_lesson_name, "\n".join(lesson_content))
                        )
                        print(
                            f"Saved lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                        )
                        lesson_content = []
                    if not current_part:
                        print(f"Error: Lesson without part in {filepath}")
                        return None
                    current_lesson_name = line[4:].strip()
                    in_code_block = False
                    print(f"Found lesson name: {current_lesson_name}")
                    continue

                if current_lesson_name and (
                    line.startswith("    ") or line.startswith("\t")
                ):
                    in_code_block = True
                    content_line = line[4:] if line.startswith("    ") else line[1:]
                    lesson_content.append(content_line)
                    print(f"Added to lesson content: '{content_line}'")
                    continue

                if in_code_block and not line.strip():
                    lesson_content.append("")
                    print(f"Added blank line to lesson content")
                    continue

                if (
                    in_code_block
                    and line.strip()
                    and not (line.startswith("    ") or line.startswith("\t"))
                ):
                    in_code_block = False
                    print("Code block end (non-indented line)")

            # Save the last lesson and part
            if current_lesson_name and lesson_content:
                current_part.lessons.append(
                    Lesson(current_lesson_name, "\n".join(lesson_content))
                )
                print(
                    f"Saved final lesson: {current_lesson_name} with content: {'\\n'.join(lesson_content)}"
                )
            if current_part:
                parts.append(current_part)
                print(f"Saved final part: {current_part_name}")

            if course_name and parts:
                print(
                    f"Created hierarchical course: {course_name} with {len(parts)} parts"
                )
                return Course(course_name, parts)
            print(f"No valid course or parts in {filepath}")
            return None
