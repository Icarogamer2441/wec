class WECError(Exception):
    """Base class for WEC language errors"""
    def __init__(self, message, line=None, file=None):
        self.message = message
        self.line = line
        self.file = file
        super().__init__(self.format_message())

    def format_message(self):
        loc = ""
        if self.file:
            loc += f" in {self.file}"
        if self.line:
            loc += f" at line {self.line}"
        return f"{self.__class__.__name__}{loc}: {self.message}"

# Common language errors
class UserInterruptError(WECError):
    """CTRL+C pressed during execution"""
    def __init__(self):
        super().__init__("Program interrupted by user")

class TypeMismatchError(WECError):
    """Type mismatch error"""
    def __init__(self, expected, got, line=None, file=None):
        super().__init__(f"Expected {expected}, got {got}", line, file)

class DivisionByZeroError(WECError):
    """Division by zero error"""
    def __init__(self, line=None, file=None):
        super().__init__("Division by zero", line, file)

class UndefinedVariableError(WECError):
    """Undefined variable error"""
    def __init__(self, name, line=None, file=None):
        super().__init__(f"Undefined variable '{name}'", line, file)

class IndexOutOfBoundsError(WECError):
    """Index out of bounds error"""
    def __init__(self, index, length, line=None, file=None):
        super().__init__(f"Index {index} out of bounds for length {length}", line, file)

class FunctionArgumentError(WECError):
    """Function argument error"""
    def __init__(self, expected, received, line=None, file=None):
        super().__init__(f"Expected {expected} arguments, got {received}", line, file)

class FileNotFoundError(WECError):
    """File not found error"""
    def __init__(self, filename, line=None, file=None):
        super().__init__(f"File '{filename}' not found", line, file)

class ImportError(WECError):
    """Module import error"""
    def __init__(self, module, line=None, file=None):
        super().__init__(f"Could not import module '{module}'", line, file)

class InvalidOperationError(WECError):
    """Invalid operation error"""
    def __init__(self, operation, types, line=None, file=None):
        type_names = " and ".join([t.__name__ for t in types])
        super().__init__(f"Invalid operation '{operation}' for {type_names}", line, file)

class KeyError(WECError):
    """Invalid key access error"""
    def __init__(self, key, line=None, file=None):
        super().__init__(f"Key '{key}' not found", line, file)

class RangeError(WECError):
    """Invalid range error"""
    def __init__(self, start, end, line=None, file=None):
        super().__init__(f"Invalid range {start}..{end}", line, file)

class MutabilityError(WECError):
    """Immutable value modification error"""
    def __init__(self, name, line=None, file=None):
        super().__init__(f"Cannot modify immutable value '{name}'", line, file)

class WECException(WECError):
    """User-throwable exception type"""
    def __init__(self, message, line=None, file=None):
        super().__init__(message, line, file) 