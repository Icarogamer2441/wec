#!/usr/bin/env python3
import sys
from parser_lexer import parse_wec
from interpreter import Interpreter, set_compiled_mode
from errors import WECError
import traceback

def main():
    version = "1.0.1"
    if len(sys.argv) < 2:
        print("Usage: wec.py [-i|-c|-x|-v|-h] <source file>")
        sys.exit(1)

    # Verifica se o modo de instalação foi passado.
    if sys.argv[1] == "-i" or sys.argv[1] == "--install":
        if len(sys.argv) < 3:
            print("Usage: wec.py -i <source.wec>")
            sys.exit(1)
        file_to_install = sys.argv[2]
        import os
        import shutil
        dest_dir = os.path.join(os.path.expanduser("~"), "weclibs")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_file = os.path.join(dest_dir, os.path.basename(file_to_install))
        try:
            shutil.copy(file_to_install, dest_file)
            print(f"File '{file_to_install}' installed at '{dest_file}'.")
        except Exception as e:
            print(f"Error copying file: {e}")
        sys.exit(0)
    elif sys.argv[1] == "-v" or sys.argv[1] == "--version":
        print("WEC Version:", version)
        sys.exit(0)
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("WEC - We Enhance Code")
        print("Usage: wec.py [-i|-c|-v|-h] <source file>")
        print("Options:")
        print("  -i <source.wec>  Install a library")
        print("  -c <source.wec>  Compile to JIT compiled mode using Numba")
        print("  -x <source.wec>  Compile to executable using Nuitka")
        print("  -v, --version    Show the version of WEC")
        print("  -h, --help       Show this help message")
        sys.exit(0)
    elif sys.argv[1] == "-c" or sys.argv[1] == "--compile":
        # --- New: Compilation mode using Numba ---
        if len(sys.argv) < 3:
            print("Usage: wec.py -c <source.wec>")
            sys.exit(1)
        filename = sys.argv[2]
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
        ast = parse_wec(code)
        try:
            import numba
        except ImportError:
            print("Error: Numba is required for compilation mode. Please install numba.")
            sys.exit(1)
        # Set the global compiled mode flag in the interpreter.
        set_compiled_mode(True)
        # Cria o interpretador e define uma função de execução compilada.
        interpreter = Interpreter()
        def run_interpreter():
            interpreter.interpret(ast)
        # Use numba.jit para "compilar" a função forçando o modo objeto, pois o código é dinâmico
        compiled_run = numba.jit(forceobj=True)(run_interpreter)
        print("Running in compiled (Numba JIT) mode …")
        try:
            compiled_run()
        except WECError as e:
            print(str(e))
        except Exception as e:
            print(f"Internal error: {str(e)}")
            if "-d" in sys.argv:
                traceback.print_exc()
        sys.exit(0)
    else:
        filename = sys.argv[1]
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
        # Faz o parse e gera a AST do código fonte.
        ast = parse_wec(code)
        # Cria o interpretador e executa a AST.
        interpreter = Interpreter()
        try:
            interpreter.interpret(ast)
        except WECError as e:
            print(str(e))
            sys.exit(1)
        except Exception as e:
            print(f"Internal error: {str(e)}")
            if "-d" in sys.argv:
                traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()
