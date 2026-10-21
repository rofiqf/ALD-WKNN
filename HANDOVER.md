# HANDOVER — Proyek ALD-WKNN (sesi ditutup 2026-10-21)

Dokumen ini ditulis untuk sesi agent baru. **Baca file ini dulu sebelum mengerjakan apa pun.**

## Status proyek: PIPELINE LENGKAP 100%

Semua tahap spec (`PROJECT_SPEC.md`) selesai dan terverifikasi. Jangan mulai ulang pipeline kecuali diminta eksplisit.

### Hasil utama (semua angka traceable ke artifact)
- Main nested CV: run `20261021T145425Z_seed20261021_main`, 480 fold-rows (8 dataset × 4 method × 15 outer fold), 0 NaN, 0 fold gagal.
- ALD-WKNN vs Power-WKNN: **5 menang, 2 seri, 1 kalah**, mean Δ Macro F1 **+0.0120**.
- Friedman: χ²(3)=7.0000, **p=0.0719**, Kendall W=0.2917 → tidak cukup bukti; post-hoc Wilcoxon/Holm **sengaja tidak dijalankan** (sesuai protokol, p≥0.05).
- Ablation: A3 full (0.901) > A0 distance-only; A1 density-only paling lemah (density berguna sebagai modifier).
- Overhead: fit 13.46× uniform KNN.
- Reproducibility check: **PASS** (atol=1e-12) via `scripts/validate_reproducibility.py --run-name main`.
- Tests: **24 passed**.

### Deliverable final
- `paper/paper.pdf` — 6 halaman, format Prosiding IEEE (IEEEtran 2 kolom), 0 error kompilasi, 13/13 sitasi.
- `paper/paper.md`, `paper/paper.tex`, `paper/references.bib` (21 entri), `paper/generated/final_audit.md` (checklist audit lengkap).
- Figure PNG+PDF di `artifacts/figures/`; tabel di `artifacts/tables/`; statistik di `artifacts/statistics/`.

### Keputusan penting yang sudah final
1. **Novelty gate CLOSED**: full-text Han 2025 (DDW-KNN, DOI 10.1145/3796159.3796172, PDF ada di `/home/ropix/3796159.3796172.pdf`) memuat gate densitas `exp(−τ|log ρ ratio|)` yang ekuivalen dengan inti ALD-WKNN. Paper sudah diposisikan sebagai **varian minimal replication/extension-style** dan mengutip Han secara terbuka. Detail: `artifacts/environment/literature_search.md`.
2. **Kepatuhan referensi** (aturan dosen: hanya Scopus/IEEE/ACM/ScienceDirect/SpringerLink): 13 referensi bernomor semuanya dari portal izin; 8 sitiran dataset UCI dipindah ke bagian **Data availability** (bukan referensi literatur).

## State teknis server (penting untuk sesi baru)

- Repo: `/home/ropix/ald-wknn-paper`, remote `https://github.com/rofiqf/ALD-WKNN` (public, branch `master`, HEAD `52761c6` — sudah terpush semua).
- **venv**: `.venv/` dibuat dengan `python3.12 -m venv --without-pip` + get-pip bootstrap, karena `ensurepip`/`python3.12-venv` TIDAK ada dan sudo butuh password. Pakai selalu `.venv/bin/python`.
- **Tools user-space di `~/bin/`** (sudah di PATH via `~/.bashrc`): `pandoc` 3.12.1, `gh` 2.103.0, plus **TinyTeX** di `~/.TinyTeX` (recompile PDF: `export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" && cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`).
- **Jaringan**: OpenML TERBLOKIR (timeout SYN-SENT — jangan coba lagi). API ucimlrepo berfungsi untuk 7 dataset; **Seeds hanya bisa via static zip UCI** (fallback sudah diimplementasi di `src/aldwknn/data.py`). GitHub, PyPI, yihui.org, CTAN bisa diakses.
- Data terdownload & tervalidasi SHA256 di `data/raw/` (8 dataset, metadata + hash tersimpan). `make download` aman dijalankan ulang.
- **gh masih terautentikasi sebagai `rofiqf`**, tetapi PAT-nya seharusnya SUDAH di-revoke user — untuk push berikutnya perlu PAT baru (`gh auth login --with-token` lalu `gh auth setup-git`).
- Cron cek berkala: sudah dihapus (tidak ada tugas background lagi).

## Cara menjalankan ulang (jika diminta)

```bash
cd /home/ropix/ald-wknn-paper
export PYTHONHASHSEED=20261021 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q                 # 24 test
./scripts/run_all.sh                          # pipeline penuh (ci+download+smoke+experiment+...)
```

Runtime aktual jauh di bawah estimasi spec: main experiment ±15 menit di server ini (4 vCPU). Run lama TIDAK ditimpa — run baru membuat `run_id` timestamp baru, symlink `artifacts/runs/latest` berpindah; analisis (stats/tables/figures/paper) mengikuti run terbaru non-ablation otomatis.

## ITEM TERBUKA (untuk user / sesi baru)

1. **`\author{Anonymous}` di `paper/paper.tex`** belum diisi nama/NIM. Isi, recompile (perintah di atas), `git commit && git push`. Minta identitas ke user bila perlu.
2. **Cek Turnitin di institusi** (target similarity ≤18%): submit HANYA `paper/paper.pdf`. Pass parafrase sudah diterapkan di template `scripts/build_paper.py`. Jika hasil >18%, minta user paste bagian yang di-flag, lalu parafrase string terkait di `build_paper.py` TEMPLATE / `make_tables.py`, regenerasi (`make tables && make paper`), recompile PDF. Turnitin tidak bisa dijalankan dari server ini.
3. **Token GitHub**: user diminta me-revoke PAT yang dipakai 2026-10-21 (sudah tereksos di chat). Verifikasi tidak perlu ditanya lagi.
4. Opsional: un-ignore `artifacts/figures/*` di `.gitignore` + push agar `paper.md` tampil gambar di GitHub.
5. Verifikasi indeks Scopus **tidak diperlukan** (aturan dosen = portal sumber, bukan indeks).

## Kontak antar-sesi

Sesi baru cukup dibrief: "Lanjutkan proyek ALD-WKNN, baca `/home/ropix/ald-wknn-paper/HANDOVER.md` dan `PROJECT_SPEC.md`." Semua keputusan desain, angka, dan provenance tersimpan di repo + artifact; tidak ada konteks yang hanya ada di sesi lama.
