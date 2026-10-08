"""document_handler/processor.py — DocumentProcessor 测试（loader 全部打桩）。"""
import asyncio

import pytest
from langchain_core.documents import Document

import app.rag.document_handler.processor as proc_module
from app.rag.document_handler.processor import DocumentProcessor


def _processor():
    return DocumentProcessor(md5_store=object())


async def _async_loader(path, *args, **kwargs):
    return [Document(page_content=f"loaded:{path}")]


def _sync_loader(path, *args, **kwargs):
    return [Document(page_content=f"loaded-sync:{path}")]


@pytest.fixture
def stub_loaders(monkeypatch):
    """把模块内引用的所有 loader 替换为打桩函数。"""
    for name in [
        "txt_loader", "pdf_loader", "markdown_loader", "ppt_loader", "word_loader",
        "pdf_multimodal_loader",
    ]:
        monkeypatch.setattr(proc_module, name, _async_loader)
    for name in [
        "txt_loader_sync", "pdf_loader_sync", "markdown_loader_sync",
        "ppt_loader_sync", "word_loader_sync", "pdf_multimodal_loader_sync",
    ]:
        monkeypatch.setattr(proc_module, name, _sync_loader)


class TestGetFileDocument:
    async def test_txt_dispatch(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "txt_loader", lambda p: (calls.append(p) or _async_loader(p)))
        docs = await _processor().get_file_document("note.txt")
        assert calls == ["note.txt"]
        assert docs[0].page_content == "loaded:note.txt"

    async def test_pdf_without_md5_uses_pdf_loader(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "pdf_loader", lambda p: (calls.append(p) or _async_loader(p)))
        docs = await _processor().get_file_document("doc.pdf")
        assert calls == ["doc.pdf"]
        assert isinstance(docs, list)

    async def test_pdf_with_md5_and_user_uses_multimodal_loader(self, stub_loaders, monkeypatch):
        calls = {}
        async def _mm(path, md5, user):
            calls["path"], calls["md5"], calls["user"] = path, md5, user
            return [Document(page_content="mm-doc")]
        monkeypatch.setattr(proc_module, "pdf_multimodal_loader", _mm)
        docs = await _processor().get_file_document("doc.pdf", "md5-1", "user-1")
        assert calls == {"path": "doc.pdf", "md5": "md5-1", "user": "user-1"}
        assert docs[0].page_content == "mm-doc"

    async def test_markdown_dispatch(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "markdown_loader", lambda p: (calls.append(p) or _async_loader(p)))
        await _processor().get_file_document("readme.md")
        assert calls == ["readme.md"]

    async def test_pptx_dispatch(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "ppt_loader", lambda p: (calls.append(p) or _async_loader(p)))
        await _processor().get_file_document("slides.pptx")
        assert calls == ["slides.pptx"]

    async def test_docx_dispatch(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "word_loader", lambda p: (calls.append(p) or _async_loader(p)))
        await _processor().get_file_document("doc.docx")
        assert calls == ["doc.docx"]

    async def test_unknown_extension_returns_empty(self, stub_loaders):
        assert await _processor().get_file_document("unknown.xyz") == []


class TestGetFileDocumentSync:
    def test_txt_dispatch_sync(self, stub_loaders, monkeypatch):
        calls = []
        monkeypatch.setattr(proc_module, "txt_loader_sync", lambda p: (calls.append(p) or _sync_loader(p)))
        docs = _processor().get_file_document_sync("note.txt")
        assert calls == ["note.txt"]
        assert docs[0].page_content == "loaded-sync:note.txt"

    def test_pdf_sync_uses_multimodal_when_md5_and_user(self, stub_loaders, monkeypatch):
        calls = {}
        def _mm(path, md5, user):
            calls.update(path=path, md5=md5, user=user)
            return [Document(page_content="mm")]
        monkeypatch.setattr(proc_module, "pdf_multimodal_loader_sync", _mm)
        docs = _processor().get_file_document_sync("doc.pdf", "md5-1", "user-1")
        assert calls == {"path": "doc.pdf", "md5": "md5-1", "user": "user-1"}
        assert docs[0].page_content == "mm"

    def test_unknown_extension_returns_empty_sync(self, stub_loaders):
        assert _processor().get_file_document_sync("unknown.xyz") == []


class TestSplitDocumentsSync:
    def test_long_document_splits_into_chunks(self):
        processor = _processor()
        doc = Document(page_content="段落内容。" * 500, metadata={"source": "a.txt"})
        chunks = processor.split_documents_sync([doc])
        assert len(chunks) > 1
        # chunk_size 来自 document_config（1000）
        assert all(len(c.page_content) <= 1000 for c in chunks)
        # 元数据被保留到每个 chunk
        assert all(c.metadata["source"] == "a.txt" for c in chunks)

    def test_short_document_stays_single_chunk(self):
        processor = _processor()
        doc = Document(page_content="短内容")
        chunks = processor.split_documents_sync([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "短内容"

    def test_empty_documents(self):
        processor = _processor()
        assert processor.split_documents_sync([]) == []


class _Upload:
    """最小 UploadFile 替身。"""

    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data

    async def seek(self, *_):
        return 0


class _Md5Store:
    """记录 save_md5_hex 调用；check 恒为未命中。"""

    def __init__(self):
        self.saved = []

    async def check_md5_hex(self, *_):
        return False

    async def save_md5_hex(self, md5, filename, original_filename, user_id):
        self.saved.append((md5, filename, original_filename, user_id))


class TestGetDocumentTempFileFlush:
    """回归：get_document 写临时文件后必须 flush + close。

    背景：tempfile.NamedTemporaryFile 默认带缓冲（8KB），小文件写完若不落盘，
    loader 用新句柄读取会得到空内容 → 切分为 0 块 → 接口仍返回"上传成功"，
    但知识库实际什么都没入库（日志只留一句「切分内容为空，跳过」）。

    本测试让 loader 复刻真实行为（另开句柄读盘），未 flush 时断言必然失败。
    """

    async def test_small_upload_content_reaches_loader(self, monkeypatch):
        # 约 480B，远小于 8KB 缓冲区——正是历史 bug 的触发区间
        payload = "RAG 检索增强生成".encode("utf-8") * 40
        seen = {}

        async def _disk_loader(path):
            # 复刻真实 loader：以新句柄读盘，验证内容确已落盘
            with open(path, "rb") as f:
                seen["raw"] = f.read()
            return [Document(page_content=seen["raw"].decode("utf-8"))]

        async def _fake_md5(_path):
            return "d41d8cd98f00b204e9800998ecf8427e"

        scheduled = []

        async def _fake_schedule(user_id, md5, filename, content, chunks=None):
            scheduled.append({"md5": md5, "filename": filename, "chunks": chunks})
            return True

        monkeypatch.setattr(proc_module, "txt_loader", _disk_loader)
        monkeypatch.setattr(proc_module, "get_file_md5_hex", _fake_md5)
        monkeypatch.setattr(
            "app.graph.services.graph_service.maybe_schedule_doc_extraction", _fake_schedule
        )

        store = _Md5Store()
        await DocumentProcessor(md5_store=store).get_document(
            files=[_Upload("small.txt", payload)], user_id="u1"
        )
        await asyncio.sleep(0)  # 放行 create_task 排入的图谱入队任务

        assert seen["raw"] == payload, "临时文件未 flush/close，loader 读到了空内容"
        assert len(store.saved) == 1
        assert len(scheduled) == 1
        assert scheduled[0]["chunks"], "应携带预切切片入队"
