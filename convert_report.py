"""
Module: convert_report.py
Mục đích: Chuyển đổi BAO_CAO_ASSIGNMENT_04.md sang BAO_CAO_ASSIGNMENT_04.html chuẩn in ấn học thuật:
1. Sửa triệt để lỗi khoảng cách chữ thưa (justify gaps): Thiết lập li { text-align: left !important; } cho toàn bộ danh sách liệt kê, p căn đều chuẩn inter-word.
2. Chuyển đổi chính xác khối mã code và inline code sang <code> và <pre><code> (không để sót dấu backtick thô).
3. Bảo vệ tuyệt đối 100% công thức toán học LaTeX (display math $$ và inline math $) không bị markdown can thiệp.
4. Không bắt buộc ngắt trang từ Mục lục sang Chương 1 (liền mạch theo dòng đọc tự nhiên).
5. Loại bỏ hoàn toàn gạch chân cho các liên kết trong Mục lục (text-decoration: none).
6. Chuẩn màu đen thuần túy (#000000) và nền trắng thuần túy (#FFFFFF), font Times New Roman, không đường kẻ ngang.
"""

import os
import re
import sys
import html as html_lib
import markdown

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def convert():
    md_path = "BAO_CAO_ASSIGNMENT_04.md"
    html_path = "BAO_CAO_ASSIGNMENT_04.html"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 1. Bảo vệ khối code và inline code
    code_blocks = []
    def save_code(m):
        raw = m.group(0)
        idx = len(code_blocks)
        if raw.startswith("```"):
            lines = raw.split("\n")
            lang = lines[0][3:].strip()
            content = "\n".join(lines[1:-1])
            escaped_content = html_lib.escape(content)
            cls_attr = f' class="language-{lang}"' if lang else ''
            rep = f'<pre><code{cls_attr}>{escaped_content}</code></pre>'
        else:
            # Inline code: `code`
            content = raw[1:-1]
            escaped_content = html_lib.escape(content)
            rep = f'<code>{escaped_content}</code>'
            
        code_blocks.append(rep)
        return f"%%CODEBLOCK_{idx}%%"

    # Match fenced code blocks ``` ... ``` and inline code ` ... `
    p_text = re.sub(r'```[\s\S]*?```|`[^`\n]+`', save_code, md_text)

    # 2. Bảo vệ Display Math $$ ... $$
    display_math = []
    def save_display_math(m):
        idx = len(display_math)
        display_math.append(m.group(0))
        return f"%%DISPLAYMATH_{idx}%%"

    p_text = re.sub(r'\$\$[\s\S]*?\$\$', save_display_math, p_text)

    # 3. Bảo vệ Inline Math $ ... $
    inline_math = []
    def save_inline_math(m):
        idx = len(inline_math)
        inline_math.append(m.group(0))
        return f"%%INLINEMATH_{idx}%%"

    p_text = re.sub(r'(?<!\$)\$(?!\$)((?:\\.|[^\$\n])+?)(?<!\$)\$(?!\$)', save_inline_math, p_text)

    # 4. Biên dịch Markdown sang HTML
    html_body = markdown.markdown(p_text, extensions=['tables', 'fenced_code', 'toc'])

    # 5. Khôi phục Inline Math
    for i, m in enumerate(inline_math):
        html_body = html_body.replace(f"%%INLINEMATH_{i}%%", m)

    # 6. Khôi phục Display Math
    for i, m in enumerate(display_math):
        html_body = html_body.replace(f"<p>%%DISPLAYMATH_{i}%%</p>", f'<div class="math-display">{m}</div>')
        html_body = html_body.replace(f"%%DISPLAYMATH_{i}%%", f'<div class="math-display">{m}</div>')

    # 7. Khôi phục Code blocks (Đã được format chuẩn thẻ <code> và <pre><code>)
    for i, c in enumerate(code_blocks):
        html_body = html_body.replace(f"%%CODEBLOCK_{i}%%", c)

    # 8. Bọc Mục lục trong container để xóa gạch chân
    def wrap_toc(match):
        toc_content = match.group(1)
        return f'<div id="table-of-contents" class="toc-container">\n{toc_content}\n</div>'

    html_body = re.sub(
        r'(<h2 id="muc-luc-chi-tiet">[\s\S]*?)(?=<h1 id="chuong-1)',
        wrap_toc,
        html_body
    )

    # Khung HTML đầy đủ chuẩn học thuật quốc tế
    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BÁO CÁO KHOA HỌC: MẠNG NƠ-RON HỒI QUY (RNN, LSTM, GRU) TRONG DỰ BÁO CHUỖI THỜI GIAN VÀ TRIỂN KHAI THỜI GIAN THỰC</title>
    
    <!-- Cấu hình MathJax 3 cho cả Display Math ($$) và Inline Math ($) -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }},
      svg: {{ fontCache: 'global' }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    
    <style>
        @page {{
            size: A4;
            margin: 20mm;
            @bottom-center {{
                content: counter(page);
                font-family: 'Times New Roman', Times, serif;
                font-size: 10pt;
                color: #000000;
            }}
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #000000 !important;
            background-color: #FFFFFF !important;
            margin: 0 auto;
            padding: 40px;
            max-width: 900px;
        }}
        hr {{ display: none !important; }}
        h1, h2, h3, h4, h5, h6 {{
            color: #000000 !important;
            font-family: 'Times New Roman', Times, serif;
            font-weight: bold;
            border-bottom: none !important;
            padding-bottom: 0 !important;
            margin-top: 1.4em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
        }}
        h1 {{ 
            font-size: 16pt; 
            text-transform: uppercase; 
            margin-top: 1.8em; 
        }}
        h2 {{ font-size: 14pt; text-transform: uppercase; margin-top: 1.5em; }}
        h3 {{ font-size: 13pt; }}
        h4 {{ font-size: 12pt; font-style: italic; }}
        
        /* CĂN ĐỀU VĂN BẢN VÀ KHẮC PHỤC KHOẢNG CÁCH THƯA */
        p {{ 
            color: #000000 !important; 
            text-align: justify; 
            text-justify: inter-word;
            margin-bottom: 0.8em; 
        }}
        /* Căn trái tự nhiên cho danh sách liệt kê để không bao giờ bị giãn khoảng cách chữ thưa */
        li {{ 
            color: #000000 !important; 
            text-align: left !important; 
            margin-bottom: 0.6em; 
            line-height: 1.6;
        }}
        ul, ol {{
            margin-bottom: 1em;
            padding-left: 28px;
        }}
        
        /* ĐỊNH DẠNG MỤC LỤC: KHÔNG GẠCH CHÂN, MÀU ĐEN THUẦN */
        .toc-container, #table-of-contents {{
            margin-bottom: 35px;
        }}
        .toc-container a, 
        #table-of-contents a, 
        #muc-luc-chi-tiet ~ ul a,
        .toc a {{
            color: #000000 !important;
            text-decoration: none !important;
        }}
        .toc-container a:hover, 
        #table-of-contents a:hover, 
        #muc-luc-chi-tiet ~ ul a:hover {{
            text-decoration: underline !important;
        }}
        .toc-container ul, #table-of-contents ul {{
            line-height: 1.8;
            padding-left: 20px;
        }}
        
        /* Khối hiển thị công thức toán học */
        .math-display {{
            text-align: center;
            margin: 15px 0;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 11pt;
            color: #000000 !important;
            background-color: #FFFFFF !important;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #000000 !important;
            padding: 8px 10px;
            text-align: left;
            color: #000000 !important;
            background-color: #FFFFFF !important;
        }}
        th {{ font-weight: bold; background-color: #FFFFFF !important; text-align: center; }}
        
        /* ĐỊNH DẠNG KHỐI MÃ NGUỒN VÀ INLINE CODE */
        pre, code {{
            font-family: 'Courier New', Courier, monospace;
            color: #000000 !important;
            background-color: #FFFFFF !important;
        }}
        pre {{
            border: 1px solid #000000;
            padding: 12px;
            overflow-x: auto;
            font-size: 10pt;
            line-height: 1.4;
            margin: 15px 0;
            page-break-inside: avoid;
            background-color: #FFFFFF !important;
        }}
        code {{ 
            font-size: 10pt; 
            padding: 1px 4px; 
            word-break: break-word;
        }}
        pre code {{
            padding: 0;
            word-break: normal;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #000000;
        }}
        blockquote {{
            margin: 15px 0;
            padding: 10px 20px;
            border-left: 3px solid #000000;
            color: #000000 !important;
            background-color: #FFFFFF !important;
            font-style: italic;
        }}
        a {{ 
            color: #000000 !important; 
            text-decoration: none; 
        }}
        
        @media print {{
            body {{ padding: 0; max-width: 100%; }}
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Đã chuyển đổi thành công sang {html_path} với căn lề chuẩn và không bị giãn thưa.")

if __name__ == "__main__":
    convert()
