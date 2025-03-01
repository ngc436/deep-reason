# deep-reason
DeepReason library for knowledge extraction

Expected input to kg pipelint is a list of Chunks with the following fields:

```json
{
    "document_id": "document id",
    "chapter_name": "name of the chapter",
    "order_id": "order id of the chunk in the document",
    "text": "chunk text",
}
```