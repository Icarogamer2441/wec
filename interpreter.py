"""
Este módulo implementa um interpretador para a linguagem wec.
Utiliza os nós da AST definidos em parser_lexer.py e executa
o código interpretado, realizando verificações de tipo simples.
"""

from parser_lexer import Program, FunctionDef, VarDecl, ReturnStmt, ExprStmt, Block, CallExpr, MemberAccess, Literal, Identifier, ListExpr, Arg, IfStmt, BinOp, StructDef, EnumDef, JoinDef, RecordLiteral, ClassDef, NewExpr, Assignment
import re

# Exceção para controlar a instrução return
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# Classe para o ambiente (escopo) do interpretador
class Environment:
    def __init__(self, outer=None):
        self.outer = outer
        self.values = {}
    def get(self, key):
        if key in self.values:
            return self.values[key]
        elif self.outer:
            return self.outer.get(key)
        else:
            raise Exception("Undefined variable: " + key)
    def set(self, key, value):
        self.values[key] = value
    def get_type(self, key):
        return self.get(key)

class EnumValue:
    def __init__(self, enum_def, variant):
        self.enum_def = enum_def
        self.variant = variant
        self.wec_type = enum_def.name  # make the type equal to the enum name

    def to_int(self):
        # Return the zero-based index of the variant in the enum's variant list
        return self.enum_def.variants.index(self.variant)

    def __repr__(self):
        return f"{self.enum_def.name}::{self.variant}"

# Função auxiliar para identificar o "tipo" de um valor
def get_type(value):
    if isinstance(value, dict) and "wec_type" in value:
        return value["wec_type"]
    elif hasattr(value, "wec_type"):
        return value.wec_type
    elif isinstance(value, bool):  # Check booleans before integers
        return "bool"
    elif isinstance(value, int):
        return "i32"
    elif isinstance(value, float):
        return "f32"
    elif isinstance(value, str):
        return "str"
    elif isinstance(value, list):
        if value:
            etype = get_type(value[0])
            return f"{etype}Array"
        else:
            return "EmptyArray"
    else:
        return type(value).__name__

def type_matches(declared, actual):
    # Exact match or if declared is like "i32Array[5]" and actual is "i32Array"
    if declared == actual:
        return True
    m = re.match(r"([0-9a-zA-Z_]+Array)\[\d+\]", declared)
    if m and m.group(1) == actual:
        return True
    return False

# ---------------------
# Implementação de Builtins
# ---------------------

# wec.print
def wec_print(*args, **kwargs):
    print(*args, **kwargs)

def wec_input(prompt):
    return input(prompt)

def wec_randomInt(a, b):
    import random
    return random.randint(a, b)

def wec_randomFloat(a, b):
    import random
    return random.uniform(a, b)

def wec_randomString(*args):
    import random
    if args:
        return random.choice(args)
    return ""

def wec_array_to_string(arr):
    return str(arr)

def wec_join_str(*args):
    return "".join(str(arg) for arg in args)

def wec_join_int(*args):
    s = "".join(str(int(arg)) for arg in args)
    try:
        return int(s)
    except:
        raise Exception("wec.join_int: cannot convert " + s + " to int")

def wec_join_float(*args):
    if len(args) == 0:
        return 0.0
    # For join_float, join the integer parts of all arguments
    # and then join the fractional parts (from 2nd arg onward)
    whole = "".join(str(int(arg)) for arg in args)
    frac = "".join((str(arg).split(".")[1] if "." in str(arg) else "") for arg in args[1:])
    if frac:
        result_str = whole + "." + frac
    else:
        result_str = whole
    try:
        return float(result_str)
    except:
        raise Exception("wec.join_float: error converting " + result_str + " to float")

def wec_type_of(val):
    return get_type(val)

# Add built-in file support

class WECFile:
    # Now accepts optional parameters so that we can later call open().
    def __init__(self, filename=None, mode=None):
        self.filename = filename
        self.mode = mode
        self.handle = None
        # Only auto-open if both filename and mode were provided.
        if filename is not None and mode is not None:
            if mode in ("w", "a"):
                self.handle = open(filename, mode)

    # Added an open() method to allow opening a file after construction.
    def open(self, filename, mode):
        self.filename = filename
        self.mode = mode
        if mode in ("w", "a"):
            self.handle = open(filename, mode)

    def write(self, text):
        if self.mode not in ("w", "a"):
            raise Exception("File not open for writing")
        if self.handle is None:
            self.handle = open(self.filename, self.mode)
        self.handle.write(text)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def read(self):
        with open(self.filename, "r") as f:
            return f.read()

class WECFileBuiltin:
    builtin_file = True
    name = "File"
    # Use *args: if no argument, create an empty file instance; if two arguments, use them.
    def new_instance(self, *args):
        if len(args) == 0:
            return WECFile()
        elif len(args) == 2:
            return WECFile(args[0], args[1])
        else:
            raise Exception("File constructor requires 0 or 2 arguments: filename and mode")
    def __call__(self, *args):
        return self.new_instance(*args)

# ---------------------
# Classe Interpretador
# ---------------------
class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self._add_builtins(self.global_env)
    def _add_builtins(self, env):
        # Cria o módulo wec com os builtins
        # "tf" será um objeto LazyTensorFlow que só importa tensorflow sob demanda
        wec_module = {
            "print": wec_print,
            "input": wec_input,
            "randomInt": wec_randomInt,
            "randomFloat": wec_randomFloat,
            "randomString": wec_randomString,
            "Array": {"to_string": wec_array_to_string},
            "join_str": wec_join_str,
            "join_int": wec_join_int,
            "join_float": wec_join_float,
            "type_of": wec_type_of,
            "File": WECFileBuiltin()  # Register the built-in File type.
        }
        env.set("wec", wec_module)
    def interpret(self, program):
        # First, process import declarations.
        if hasattr(program, "imports"):
            for import_decl in program.imports:
                self.process_import(import_decl)

        # Register type definitions, if any.
        if hasattr(program, "types"):
            for type_def in program.types:
                self.global_env.set(type_def.name, type_def)
        for func in program.functions:
            self.global_env.set(func.name, func)
        # Look for the main function
        try:
            main_func = self.global_env.get("main")
        except Exception:
            raise Exception("Main function not defined.")
        result = self.call_function(main_func, [])
        # Opcional: retorna ou imprime o valor retornado pela main
        # print("Resultado da main:", result)
    def call_function(self, func, args):
        if not isinstance(func, FunctionDef):
            raise Exception("Attempt to call a non-function: " + str(func))
        if len(args) != len(func.params):
            raise Exception("Incorrect number of arguments in call to " + func.name)
        local_env = Environment(outer=self.global_env)
        # Add the function itself to the local environment so it can be called recursively.
        local_env.set(func.name, func)
        for param, arg in zip(func.params, args):
            local_env.set(param.name, arg)
        try:
            self.execute_block(func.body, local_env)
        except ReturnException as ret:
            if func.return_type and (func.return_type != get_type(ret.value)):
                raise Exception(f"Return type mismatch in function {func.name}: declared {func.return_type}, but got {get_type(ret.value)}")
            return ret.value
        return None
    def execute_block(self, block, env):
        for stmt in block.statements:
            self.execute(stmt, env)
    def execute(self, stmt, env):
        if isinstance(stmt, VarDecl):
            if stmt.initializer is None:
                if stmt.type.endswith("List"):
                    value = WECList()
                    value.wec_type = stmt.type
                else:
                    value = None
            else:
                value = self.evaluate(stmt.initializer, env)
            if value is not None and not type_matches(stmt.type, get_type(value)):
                raise Exception(f"Type error in variable '{stmt.name}': declared {stmt.type}, but got {get_type(value)}")
            env.set(stmt.name, value)
        elif isinstance(stmt, ReturnStmt):
            value = self.evaluate(stmt.expression, env)
            raise ReturnException(value)
        elif isinstance(stmt, ExprStmt):
            self.evaluate(stmt.expression, env)
        elif isinstance(stmt, IfStmt):
            if self.evaluate(stmt.condition, env):
                self.execute_block(stmt.then_block, env)
            else:
                executed = False
                for (cond, block) in stmt.elif_blocks:
                    if self.evaluate(cond, env):
                        self.execute_block(block, env)
                        executed = True
                        break
                if not executed and stmt.else_block is not None:
                    # Se o else_block é um Block, usa execute_block; se for um IfStmt, usa execute.
                    if isinstance(stmt.else_block, Block):
                        self.execute_block(stmt.else_block, env)
                    else:
                        self.execute(stmt.else_block, env)
        elif isinstance(stmt, Assignment):
            # Execute the assignment by evaluating the right-hand side.
            if hasattr(stmt.target, "index") and stmt.target.__class__.__name__ == "IndexExpr":
                array_obj = self.evaluate(stmt.target.receiver, env)
                idx = self.evaluate(stmt.target.index, env)
                array_obj[idx] = self.evaluate(stmt.value, env)
            elif isinstance(stmt.target, MemberAccess):
                val = self.evaluate(stmt.value, env)
                receiver = self.evaluate(stmt.target.receiver, env)
                if isinstance(receiver, dict):
                    receiver[stmt.target.member] = val
                elif isinstance(receiver, Instance):
                    receiver.fields[stmt.target.member] = val  # <-- update Instance fields
                else:
                    setattr(receiver, stmt.target.member, val)
            else:
                raise Exception("Invalid assignment target")
        else:
            raise Exception("Instrução desconhecida: " + str(stmt))
    def evaluate(self, expr, env):
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Identifier):
            return env.get(expr.name)
        elif isinstance(expr, ListExpr):
            return [self.evaluate(el, env) for el in expr.elements]
        elif isinstance(expr, MemberAccess):
            return self.evaluate_member_access(expr, env)
        elif isinstance(expr, CallExpr):
            return self.evaluate_call_expr(expr, env)
        elif isinstance(expr, BinOp):
            left = self.evaluate(expr.left, env)
            right = self.evaluate(expr.right, env)
            op = expr.op
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right
            elif op == '%':
                return left % right
            elif op == '<=':
                return left <= right
            elif op == '<':
                return left < right
            elif op == '>=':
                return left >= right
            elif op == '>':
                return left > right
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right
            else:
                raise Exception("Unknown binary operator: " + op)
        elif isinstance(expr, RecordLiteral):
            record = {}
            for key, field_expr in expr.fields.items():
                record[key] = self.evaluate(field_expr, env)
            record["wec_type"] = expr.type_name
            # If the record's type has an implementation, attach its methods.
            try:
                type_def = env.get(expr.type_name)
                if hasattr(type_def, "methods"):
                    for method in type_def.methods:
                        record[method.name] = method
            except Exception:
                pass
            return record
        elif isinstance(expr, NewExpr):
            return self.evaluate_new_expr(expr, env)
        elif expr.__class__.__name__ == "IndexExpr":
            array_obj = self.evaluate(expr.receiver, env)
            index = self.evaluate(expr.index, env)
            return array_obj[index]
        elif isinstance(expr, FunctionDef) or hasattr(expr, 'params'):
            # Wrap the function definition so that it becomes a Python callable.
            return lambda *args: self.call_function(expr, args)
        else:
            raise Exception("Unknown expression: " + str(expr))

    def evaluate_new_expr(self, node, env):
        """
        Avalia uma expressão 'new'. Obtém a definição de classe a partir do ambiente
        de tipos global (supondo que ele esteja disponível via, por exemplo, env.get_type).
        Cria uma instância e, se houver um 'constructor', invoca-o e armazena o resultado
        no campo 'call' da instância.
        """
        # If the new expression node has an extra attribute 'member', combine it with type_name.
        if hasattr(node, "member") and node.member:
            node.type_name = f"{node.type_name}::{node.member}"

        # Now check if the (possibly modified) type name is qualified.
        if "::" in node.type_name:
            module_name, type_name = node.type_name.split("::", 1)
            module_def = env.get(module_name)
            if not isinstance(module_def, dict):
                raise Exception(f"Module '{module_name}' not found or not a module.")
            type_def = module_def.get(type_name)
            if type_def is None:
                raise Exception(f"Type '{type_name}' not found in module '{module_name}'.")
        else:
            type_def = env.get(node.type_name)
            if isinstance(type_def, dict):
                raise Exception(
                    f"Cannot instantiate module '{node.type_name}'. "
                    f"Use member access (e.g., '{node.type_name}::TypeName') to instantiate a type from the module."
                )

        # Special handling for built-in types (like File)
        if hasattr(type_def, "builtin_file") and type_def.builtin_file:
            args = []
            if hasattr(node, "args") and node.args:
                args = [self.evaluate(arg.value, env) for arg in node.args]
            elif hasattr(node, "arguments") and node.arguments:
                args = [self.evaluate(arg.value, env) for arg in node.arguments]
            elif hasattr(node, "arg_list") and node.arg_list:
                args = [self.evaluate(arg.value, env) for arg in node.arg_list]
            if len(args) not in (0, 2):
                raise Exception("File constructor requires either 0 or 2 arguments: filename and mode")
            if len(args) == 0:
                return type_def.new_instance()
            else:
                return type_def.new_instance(*args)

        # Fallback to normal interpreter instantiation for other types.
        instance = Instance(type_def)
        # If using a qualified type name, update the instance type accordingly.
        if "::" in node.type_name:
            instance.wec_type = node.type_name

        # Initialize public fields from the class definition.
        for member in type_def.members:
            if isinstance(member, tuple) and member[0] == "field":
                # member is ("field", fieldName, fieldType)
                instance.fields[member[1]] = None
        # Look for a constructor method (named "constructor") in the definition.
        constructor = instance.get_method("constructor")
        # Only call the constructor immediately if arguments were provided.
        if constructor is not None and node.args:
            local_env = Environment(outer=env)
            local_env.set("this", instance)
            if len(constructor.params) != len(node.args):
                raise Exception("Número de argumentos incompatível no construtor")
            for param, arg in zip(constructor.params, node.args):
                local_env.set(param.name, self.evaluate(arg.value, env))
            result = self.execute_block(constructor.body, local_env)
            instance.fields["call"] = result
        else:
            instance.fields["call"] = None
        return instance

    def instantiate_class(self, class_def, args):
        # Create a new instance (dictionary) with the class name as its type.
        instance = {"wec_type": class_def.name}
        # Process class members: install fields (defaulting to None) and methods.
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "field":
                # member is ("field", fieldName, fieldType)
                field_name = member[1]
                instance[field_name] = None
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "method":
                method_def = member[1]
                # Bind method to instance: create a closure that sets 'this' to instance.
                def bound_method(*m_args, method_def=method_def):
                    local_env = Environment(outer=self.global_env)
                    local_env.set("this", instance)
                    if len(method_def.params) != len(m_args):
                        raise Exception("Incorrect number of arguments in method call: " + method_def.name)
                    for param, arg in zip(method_def.params, m_args):
                        local_env.set(param.name, arg)
                    try:
                        self.execute_block(method_def.body, local_env)
                    except ReturnException as ret:
                        return ret.value
                # Attach the bound method to the instance.
                instance[method_def.name] = bound_method
        # Look for a constructor method (named "constructor") if present.
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "method" and member[1].name == "constructor":
                constructor = member[1]
                local_env = Environment(outer=self.global_env)
                local_env.set("this", instance)
                if len(constructor.params) != len(args):
                    raise Exception("Incorrect number of arguments in constructor of " + class_def.name)
                for param, arg in zip(constructor.params, args):
                    local_env.set(param.name, arg)
                try:
                    self.execute_block(constructor.body, local_env)
                except ReturnException as ret:
                    pass
                break
        return instance

    def evaluate_member_access(self, node, env):
        receiver = self.evaluate(node.receiver, env)
        if isinstance(receiver, EnumDef):
            # If the receiver is an enum definition, look for the variant.
            if hasattr(receiver, "variants") and (node.member in receiver.variants):
                return EnumValue(receiver, node.member)
            else:
                raise Exception(f"Enum variant '{node.member}' not found in enum {receiver.name}")
        elif isinstance(receiver, Instance):
            if node.member in receiver.fields:
                return receiver.fields[node.member]
            else:
                method = receiver.get_method(node.member)
                if method is not None:
                    return lambda *args: self.call_method(receiver, method, args, env)
                raise Exception(f"Method or field '{node.member}' not found in instance of {receiver.class_def.name}")
        elif isinstance(receiver, dict):
            if node.member in receiver:
                return receiver[node.member]
            else:
                method = receiver.get_method(node.member)
                if method is not None:
                    return lambda *args: self.call_method(receiver, method, args, env)
                raise Exception(f"Method or field '{node.member}' not found in dictionary")
        elif isinstance(receiver, (int, float, str)):
            if node.member == "to_str":
                return lambda *args: str(receiver)
            elif node.member == "to_int":
                return lambda *args: int(receiver)
            elif node.member == "to_float":
                return lambda *args: float(receiver)
            else:
                raise Exception(f"Member '{node.member}' not found in primitive type {type(receiver).__name__}")
        else:
            return getattr(receiver, node.member)
    
    def call_method(self, instance, method, args, env):
        local_env = Environment(outer=env)
        local_env.set("this", instance)
        # Inject module-level definitions if the instance's type is qualified (e.g., "math::Math")
        if "::" in instance.wec_type:
            module_name, _ = instance.wec_type.split("::", 1)
            try:
                module_def = self.global_env.get(module_name)
            except Exception:
                module_def = None
            if module_def and isinstance(module_def, dict):
                for key, value in module_def.items():
                    # Avoid overwriting existing definitions in the local environment.
                    if key not in local_env.values:
                        local_env.set(key, value)
        for param, arg in zip(method.params, args):
            local_env.set(param.name, arg)
        try:
            return self.execute_block(method.body, local_env)
        except ReturnException as ret:
            return ret.value

    def evaluate_call_expr(self, node, env):
        callee = self.evaluate(node.callee, env)
        args = [self.evaluate(arg.value, env) for arg in node.args]
        if isinstance(callee, Instance):
            # If an instance is called, use its 'constructor' method (if defined) as its call operator.
            method = callee.get_method("constructor")
            if method is not None:
                return self.call_method(callee, method, args, env)
            else:
                raise Exception("Instance not callable: no constructor method found")
        elif isinstance(callee, FunctionDef):
            return self.call_function(callee, args)
        elif callable(callee):
            return callee(*args)
        else:
            raise Exception("Attempt to call a non-function: " + str(callee))

    def process_import(self, import_decl):
        from parser_lexer import parse_wec
        import os, sys
        filename = import_decl.filename
        # Determine the full path based on the filename prefix.
        if filename.startswith("./"):
            if len(sys.argv) > 1:
                base_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
            else:
                base_dir = os.getcwd()
            filename = filename[2:]
            full_path = os.path.join(base_dir, filename)
        else:
            base_dir = os.path.join(os.path.expanduser("~"), "weclibs")
            full_path = os.path.join(base_dir, filename)

        if not full_path.endswith(".wec"):
            full_path += ".wec"

        if not os.path.exists(full_path):
            raise Exception(f"Imported file not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
        imported_ast = parse_wec(code)

        # Build a module namespace dictionary with the module's definitions.
        module_namespace = {}
        for t in imported_ast.types:
            module_namespace[t.name] = t
        for f in imported_ast.functions:
            module_namespace[f.name] = f

        # Determine the module name from the module declaration (if any)
        if imported_ast.module is not None:
            mod_name = imported_ast.module.name
        else:
            mod_name = os.path.splitext(os.path.basename(filename))[0]

        # Register the imported module under its name to the global environment.
        self.global_env.set(mod_name, module_namespace)

# Nova classe para representar instâncias de classes
class Instance:
    def __init__(self, class_def):
        self.class_def = class_def  # A definição da classe (ex: ClassDef)
        self.fields = {}  # Campos armazenados na instância
        self.wec_type = class_def.name  # Define the type name as the class name

    def get_method(self, name):
        # Procura por um método com o nome solicitado na definição da classe.
        for member in self.class_def.members:
            # Cada membro de método vem na forma ("method", FunctionDef)
            if isinstance(member, tuple) and member[0] == "method":
                method = member[1]
                if method.name == name:
                    return method
        return None

class WECList(list):
    def length(self):
        return len(self)

    def foreach(self, func):
        for item in self:
            func(item)

    def map(self, func):
        newList = WECList()
        newList.wec_type = self.wec_type
        for item in self:
            newList.append(func(item))
        return newList

    def filter(self, func):
        newList = WECList()
        newList.wec_type = self.wec_type
        for item in self:
            if func(item):
                newList.append(item)
        return newList
