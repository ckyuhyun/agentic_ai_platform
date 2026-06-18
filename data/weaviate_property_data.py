from dataclasses import dataclass


@dataclass
class WeaviateProperty:
    postgres_chunk_id : str = ""
    document_id : str = ""
    name : str  = ""
    content : str = ""
    datatype : any = None 
    description : str =""
    category : str = ""

