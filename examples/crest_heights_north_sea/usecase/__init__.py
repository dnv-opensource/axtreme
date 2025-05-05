from pathlib import Path
import sys

"""
We want to be able to run the content of this directory in interactive mode, and run it as a module imported using
`from usecase import <blah>` etc. The current approach means code in `usecase` can be written as if `usecase` is the
root directory. This init mean when it is imported as a module all the imports still work.

Limitations:
- Modules in this directory can conflict/override your other imports. For example you have 'usecase/math.py` this will
  override imports of the standard `math` module.
    - Comment: This will be an annoying to find bug
    - Minimisation: files here should have reasonably unique names, so should be unlikely to conflict with other root
      imports
"""
sys.path = [str(Path(__file__).parent.resolve()), *sys.path]  # this solve the issues
