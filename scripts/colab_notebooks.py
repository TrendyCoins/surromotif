"""Colab notebooks of the Colab scripts: scripts/<name>.ipynb — a text cell (the title and the run section of the docstring)
and one code cell, the .py file as is (the settings block НАСТРОЙКИ is at its top). In Colab: File -> Upload notebook.

    python scripts/colab_notebooks.py      # writes scripts/f1h_tf_line21.ipynb, scripts/f1h_tf_pack.ipynb

The .py stays the source; tests/test_f1h_tf.py checks that the notebooks are in sync with it.
"""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = ('f1h_tf_line21', 'f1h_tf_pack')


def _lines(text):
    out = text.splitlines(keepends=True)
    if out:
        out[-1] = out[-1].rstrip('\n')
    return out


def notebook(name):
    """The notebook of scripts/<name>.py as a dict (nbformat 4)."""
    with open(os.path.join(ROOT, 'scripts', name + '.py'), encoding='utf-8') as fh:
        src = fh.read()
    doc = ast.get_docstring(ast.parse(src)) or ''
    title = doc.splitlines()[0] if doc else name
    run = doc[doc.index('ЗАПУСК (Colab).'):] if 'ЗАПУСК (Colab).' in doc else ''
    text = (f"# {name}\n\n{title}\n\n{' '.join(run.split())}\n\nНастройки — блок НАСТРОЙКИ в начале ячейки ниже; "
            f"источник — scripts/{name}.py репо (блокнот собран из него, править — там).")
    return {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': _lines(text)},
                      {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': _lines(src)}],
            'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 0}


def main():
    for name in NOTEBOOKS:
        p = os.path.join(ROOT, 'scripts', name + '.ipynb')
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(notebook(name), fh, ensure_ascii=False, indent=1)
            fh.write('\n')
        print(p)


if __name__ == '__main__':
    main()
