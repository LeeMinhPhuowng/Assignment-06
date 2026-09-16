"""
Script chuyển đổi Báo cáo Markdown sang HTML chuẩn in ấn học thuật
Bảo vệ tuyệt đối 100% công thức toán học LaTeX không bị markdown làm hỏng.
Cấu hình MathJax 3 nhận diện cả inlineMath ($...$) và displayMath ($$...$$).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import re
import markdown

with open('BAO_CAO_ASSIGNMENT_04.md', 'r', encoding='utf-8') as f:
    text = f.read()

# BƯỚC 1: Trích xuất và bảo vệ công thức toán học (Math Protection)
math_blocks = []
def save_block_math(match):
    math_blocks.append(match.group(0))
    return f"@@MATH_BLOCK_{len(math_blocks)-1}@@"

# Bảo vệ khối display math $$...$$
text = re.sub(r'\$\$(.*?)\$\$', save_block_math, text, flags=re.DOTALL)

# Bảo vệ khối inline math $...$
math_inlines = []
def save_inline_math(match):
    math_inlines.append(match.group(0))
    return f"@@MATH_INLINE_{len(math_inlines)-1}@@"

text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline_math, text)

from markdown.extensions.toc import slugify_unicode

# BƯỚC 2: Chuyển đổi Markdown sang HTML (bật toc với slugify_unicode để link Mục lục khớp 100% tiếng Việt)
html_content = markdown.markdown(
    text,
    extensions=['tables', 'fenced_code', 'toc'],
    extension_configs={'toc': {'slugify': slugify_unicode}}
)

# BƯỚC 3: Khôi phục lại toàn bộ công thức toán học nguyên bản
for i, m in enumerate(math_blocks):
    html_content = html_content.replace(f"@@MATH_BLOCK_{i}@@", m)

for i, m in enumerate(math_inlines):
    html_content = html_content.replace(f"@@MATH_INLINE_{i}@@", m)

template = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo Cáo Nghiên Cứu Mạng Nơ-ron Tích Chập (Assignment 04)</title>
<!-- Cấu hình MathJax 3 hỗ trợ cả $...$ và $$...$$ -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @page {
        size: A4;
        margin: 20mm;
    }
    *, *::before, *::after {
        box-sizing: border-box;
    }
    html, body {
        max-width: 100%;
        overflow-x: hidden;
    }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 13pt;
        line-height: 1.6;
        color: #000000;
        background-color: #FFFFFF;
        max-width: 920px;
        margin: 0 auto;
        padding: 40px 25px;
        text-align: justify;
    }
    h1 {
        font-size: 18pt;
        font-weight: bold;
        text-align: center;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 25px;
        line-height: 1.4;
        color: #000000;
    }
    h3 {
        font-size: 14pt;
        font-weight: bold;
        margin-top: 32px;
        margin-bottom: 14px;
        text-transform: uppercase;
        color: #000000;
    }
    h4 {
        font-size: 13pt;
        font-weight: bold;
        margin-top: 22px;
        margin-bottom: 10px;
        color: #000000;
    }
    p, li {
        font-size: 13pt;
        margin-bottom: 12px;
        color: #000000;
    }
    a {
        color: #000000;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    ul, ol {
        margin-top: 5px;
        margin-bottom: 15px;
        padding-left: 30px;
    }
    li {
        margin-bottom: 8px;
    }
    table {
        width: 100%;
        max-width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 12pt;
        word-break: break-word;
    }
    th, td {
        border: 1px solid #000000;
        padding: 8px 12px;
        text-align: center;
        color: #000000;
    }
    th {
        font-weight: bold;
        background-color: #FFFFFF;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 25px auto;
        border: 1px solid #000000;
    }
    code {
        font-family: 'Courier New', Courier, monospace;
        font-size: 11pt;
        background-color: #FFFFFF;
        color: #000000;
        padding: 2px 4px;
        border: 1px solid #CCCCCC;
    }
    pre {
        background-color: #FFFFFF;
        border: 1px solid #000000;
        padding: 12px;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Courier New', Courier, monospace;
        font-size: 10.5pt;
        line-height: 1.4;
        max-width: 100%;
        overflow: visible;
    }
    .MathJax, mjx-container {
        max-width: 100% !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
    }
    .toc-separator, .page-break {
        display: block;
        page-break-after: always;
        break-after: page;
        margin-top: 40px;
        margin-bottom: 110px;
        border: none;
    }
    @media print {
        * {
            overflow: visible !important;
        }
        ::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        pre {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow: visible !important;
        }
        table {
            page-break-inside: avoid;
            break-inside: avoid;
        }
        h1, h2, h3, h4 {
            page-break-after: avoid;
            break-after: avoid;
        }
        .toc-separator, .page-break {
            page-break-after: always;
            break-after: page;
            margin: 0;
            padding: 0;
            height: 0;
        }
    }
    pre code {
        border: none;
        padding: 0;
    }
</style>
</head>
<body>
""" + html_content + """
</body>
</html>
"""

with open('BAO_CAO_ASSIGNMENT_04.html', 'w', encoding='utf-8') as f:
    f.write(template)

print("Đã biên dịch thành công BAO_CAO_ASSIGNMENT_04.html với bảo vệ công thức toán học tuyệt đối!")
