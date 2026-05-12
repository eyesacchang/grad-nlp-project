PAPER = main

.PHONY: all figs clean verify

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex sections/*.tex refs.bib figs/fig_main_p1.pdf figs/fig_ablation.pdf
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error $(PAPER).tex

figs:
	cd figs && python make_figures.py

clean:
	latexmk -C
	rm -f *.bbl *.blg

verify: $(PAPER).pdf
	@echo "=== Page count ==="
	@pdfinfo $(PAPER).pdf | grep -E "Pages|Page size"
	@echo ""
	@echo "=== Em dashes (target <= 3) ==="
	@grep -oE -- '—' sections/*.tex | wc -l | tr -d ' '
	@echo ""
	@echo "=== LLM tells (should be empty) ==="
	@! grep -inE 'delve|harness|underscore|pivotal|paramount|notably,|in recent years|it is worth noting' sections/*.tex
	@echo ""
	@echo "=== Anonymous compliance (should be empty) ==="
	@! pdftotext $(PAPER).pdf - 2>/dev/null | grep -iE "(isaac chang|ingrid chien|minh tran|harvard|@gmail|@harvard|@mit\.edu)"
	@echo ""
	@echo "=== Overfull boxes (should be empty) ==="
	@! grep "Overfull" $(PAPER).log
