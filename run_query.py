from config import set_global_seed
from rag.pipeline import PDFRAG

set_global_seed()    # Seed für maximal reproduzierbare Antworten
print(PDFRAG().query(input('Q: ')))