#!/usr/bin/env python3
import sys
from parser_lexer import parse_wec
from interpreter import Interpreter

def main():
    if len(sys.argv) < 2:
        print("Uso: wec.py [-i] <arquivo fonte>")
        sys.exit(1)

    # Verifica se o modo de instalação foi passado.
    if sys.argv[1] == "-i":
        if len(sys.argv) < 3:
            print("Uso: wec.py -i <arquivo.wec>")
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
            print(f"Arquivo '{file_to_install}' instalado em '{dest_file}'.")
        except Exception as e:
            print(f"Erro ao copiar o arquivo: {e}")
        sys.exit(0)
    else:
        filename = sys.argv[1]
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
        # Faz o parse e gera a AST do código fonte.
        ast = parse_wec(code)
        # Cria o interpretador e executa a AST.
        interpreter = Interpreter()
        interpreter.interpret(ast)

if __name__ == '__main__':
    main()
