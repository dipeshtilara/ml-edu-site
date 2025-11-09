# convert_prints_to_st.py
# Usage: python convert_prints_to_st.py
# Scans .py files under streamlit_app/ and replaces print(...) with st.write(...).
# It also inserts "import streamlit as st" where needed.
# BACKUP before running (commit), because this rewrites files.

import ast
from pathlib import Path
import sys

PROJECT_DIR = Path("streamlit_app")  # change if your app files are elsewhere
if not PROJECT_DIR.exists():
    print(f"Directory {PROJECT_DIR} not found. Update PROJECT_DIR in this script to your Streamlit app folder.")
    sys.exit(1)

def transform_source(src: str):
    tree = ast.parse(src)
    changed_flag = False

    class PrintRewriter(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal changed_flag
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                new_node = ast.copy_location(
                    ast.Call(
                        func=ast.Attribute(value=ast.Name(id="st", ctx=ast.Load()), attr="write", ctx=ast.Load()),
                        args=node.args,
                        keywords=node.keywords
                    ),
                    node
                )
                changed_flag = True
                return ast.fix_missing_locations(new_node)
            return self.generic_visit(node)

    new_tree = PrintRewriter().visit(tree)
    if changed_flag:
        try:
            new_src = ast.unparse(new_tree)
        except Exception:
            # fallback naive replace (rare)
            new_src = src.replace("print(", "st.write(")
        return new_src, True
    return src, False

def ensure_streamlit_import(src: str):
    if "import streamlit as st" in src:
        return src, False

    tree = ast.parse(src)
    # place import after module docstring if present
    if len(tree.body) > 0 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        # find end line of docstring
        try:
            end_line = tree.body[0].end_lineno
        except Exception:
            end_line = tree.body[0].lineno
        lines = src.splitlines(True)
        insert_at = sum(len(lines[i]) for i in range(end_line))
        new_src = src[:insert_at] + "\nimport streamlit as st\n" + src[insert_at:]
    else:
        new_src = "import streamlit as st\n" + src
    return new_src, True

modified_files = []
for p in PROJECT_DIR.rglob("*.py"):
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Skipping {p}: cannot read ({e})")
        continue

    new_text, changed = transform_source(text)
    if changed:
        new_text, imp_added = ensure_streamlit_import(new_text)
        p.write_text(new_text, encoding="utf-8")
        modified_files.append(str(p))
        print(f"Modified {p} (print -> st.write). Added import: {imp_added}")

print("\nDone. Modified files:")
for f in modified_files:
    print(" -", f)

if not modified_files:
    print("No print(...) calls found in streamlit_app/ or no modifications made.")
