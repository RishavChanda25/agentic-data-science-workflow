from langchain_experimental.utilities import PythonREPL
from typing import Dict, Any

class DataScienceREPL:
    def __init__(self):
        # Initialize the LangChain experimental REPL
        self.repl = PythonREPL()
        
    def execute_code(self, code_string: str) -> Dict[str, Any]:
        """
        Executes a string of Python code and captures the output or errors.
        """
        import textwrap
        
        clean_code = self._clean_markdown(code_string)
        
        # Forcefully wrap the LLM's code in a try-except block to guarantee error capture
        indented_code = textwrap.indent(clean_code, "    ")
        wrapped_code = f"""
try:
{indented_code}
except Exception as e:
    import traceback
    print("CRITICAL_EXECUTION_ERROR")
    print(traceback.format_exc())
"""
        
        try:
            result = self.repl.run(wrapped_code)
            
            # 1. Check for our guaranteed error flag
            if "CRITICAL_EXECUTION_ERROR" in result:
                # Strip our flag and return the raw traceback to the LLM
                clean_error = result.replace("CRITICAL_EXECUTION_ERROR", "").strip()
                return {
                    "success": False,
                    "output": clean_error
                }
            
            # 2. If no error flag, it succeeded
            if not result.strip():
                result = "Code executed successfully with no output."
                
            return {
                "success": True,
                "output": result.strip()
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"System execution failed with error: {str(e)}"
            }
            
    def _clean_markdown(self, text: str) -> str:
        """Helper to remove markdown code blocks so Python's exec() doesn't choke."""
        if "```python" in text:
            text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]
        return text.strip()