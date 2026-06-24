"""定义保存历史对话到 Markdown 文件的函数"""
from datetime import datetime
import os

def save_conversation(user_input, answer):

    os.makedirs("memory", exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    md_file = f"memory/{today}.md"

    current_time = datetime.now().strftime("%H:%M:%S")

    with open(md_file, "a", encoding="utf-8") as f:

        f.write(f"\n# {current_time}\n\n")

        f.write("## User\n")
        f.write(user_input + "\n\n")

        f.write("## Assistant\n")
        f.write(answer + "\n\n")

        f.write("---\n")