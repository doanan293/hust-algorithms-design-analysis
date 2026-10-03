# Report LaTeX

Báo cáo môn Design and Analysis of Algorithms, đề số 5, viết bằng tiếng Anh theo cấu trúc một bài báo. Article một cột, pdflatex.

## Build

Dùng script của skill `latex-document-skill` (tự chạy đủ lượt pdflatex + biber, dọn file tạm, xuất PNG xem nhanh):

```bash
cd docs/report
S=../../.claude/skills/latex-document-skill/scripts
bash $S/compile_latex.sh main.tex --preview --preview-dir build/preview   # -> main.pdf, build/preview/main-N.png
bash $S/compile_latex.sh main.tex --verbose                                # xem đầy đủ log khi có lỗi
bash $S/compile_latex.sh main.tex --clean                                  # xoá file tạm
```

Yêu cầu: TeX Live theo danh sách gói của skill (`scripts/install_deps.sh`) cộng `biber`, `latexmk`, `poppler-utils`.

VS Code: extension **LaTeX Workshop** dùng `latexmk` mặc định, mở `main.tex` và nhấn Build là được; PDF nằm cạnh `main.tex`.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| `main.tex` | Tiêu đề, tác giả, ghép các phần. Không viết nội dung ở đây. |
| `preamble.tex` | Gói, tiếng Anh (T1 + babel english), định lý, giả mã, tham chiếu chéo. |
| `sections/*.tex` | Abstract, Introduction, Related Work, System Model, Problem Analysis, Proposed Method, Experimental Evaluation, Discussion, Conclusions, Appendix. |
| `tables/*.tex` | Bảng tách file, `\input` từ section. |
| `figures/` | Ảnh (`\includegraphics{ten-file}` không cần đuôi). |
| `references.bib` | Tài liệu tham khảo; trích dẫn bằng `\cite{key}`. |

## Quy ước viết

- **Tham chiếu chéo:** `\cref{fig:x}` → "Figure 3", `\Cref{sec:x}` ở đầu câu. Nhãn: `sec:`, `fig:`, `tab:`, `eq:`, `alg:`, `thm:`.
- **Trích dẫn:** `\cite{huang2025ddatsap}`; tài liệu được đánh số theo thứ tự xuất hiện (style IEEE).
- **Định lý:** `theorem`, `lemma`, `proposition`, `definition`, `remark` (tên tiếng Anh).
- **Giả mã:** môi trường `algorithm` + `algorithmic`; "Đầu vào/Đầu ra" qua `\Require`/`\Ensure`.
- **Đơn vị:** `\qty{100}{\metre}`, `\qty{-174}{\dBm\per\hertz}` (siunitx).
- **Hình:** `width=0.75–0.85\textwidth`, vị trí `[htbp]`; chỉ dùng `[H]` khi bắt buộc.
- **Văn bản:** ưu tiên đoạn văn hơn bullet; ký tự `<`, `>` phải viết trong `$...$`; `%`, `&`, `_`, `#` phải escape.
- Nội dung trong `[ngoặc vuông]` là chỗ cần điền. Không bắt đầu nội dung `proof`/`theorem` bằng `[` vì LaTeX hiểu là tuỳ chọn tên.
- Số thập phân dùng dấu chấm (siunitx mặc định). Nhãn chữ trong `data/*.csv` do các script `experiments/report_tables*.py` sinh bằng tiếng Anh; riêng `data/related_work.csv` viết tay.

## Công cụ thêm từ skill

```bash
S=../../.claude/skills/latex-document-skill/scripts
bash $S/latex_lint.sh main.tex                                          # chktex
python3 $S/generate_chart.py line --data '{...}' --output figures/x.png  # vẽ chart matplotlib
```
Hướng dẫn viết tài liệu dài: `.claude/skills/latex-document-skill/references/long-form-best-practices.md`.
