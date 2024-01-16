import uuid
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from toolboxv2.utils.cryp import Code


def get_id(name=str(uuid.uuid4())):
    return Code.one_way_hash(name, "quickNote", "id-generator")


class DataTypes(Enum):
    MD: str = "MarkDown"
    HTML: str = "HTML"
    TEXT: str = "TEXT"
    CUSTOM: str = "CUSTOM"


class Tag(BaseModel):
    name: str
    id: str
    related: list[str]

    @classmethod
    def crate(cls, name: str, related=None):
        if related is None:
            related = []
        return cls(name=name, id=get_id(name), related=related)

    @classmethod
    def crate_root(cls):
        return cls(name="root", id=get_id("root"), related=[])

    def add_related(self, other_tag):
        self.related.append(other_tag.id)

    def print(self, show=True, debug=False, data=False):
        string_data = f"= {self.name} ="
        if debug:
            string_data += f"_________ debug Data _________{self.id}-{self.related} "
        for r_id in self.related:
            string_data += r_id if debug else ""
        if data:
            string_data = {
                "id": self.id,
                "name": self.name,
                "related": self.related,
            }
        if show:
            print(string_data)
        else:
            return string_data


class Note(BaseModel):
    name: str
    id: str
    data: str
    data_type: DataTypes
    tags: list[Tag]
    links: list[str]
    parent: Tag

    @classmethod
    def crate_root(cls):
        root_tag = Tag.crate_root()
        return cls(id=get_id("root"),
                   name="root",
                   data="",
                   data_type=DataTypes.TEXT,
                   tags=[root_tag],
                   links=[],
                   parent=root_tag)

    @classmethod
    def crate_new_text(cls, name: str, data: str, tag: Tag, parent: Tag or None = None, links: list or None = None):
        if parent is None:
            parent = tag
        if links is None:
            links = []
        return cls(id=get_id(name),
                   name=name,
                   data=data,
                   data_type=DataTypes.TEXT,
                   tags=[tag],
                   links=links,
                   parent=parent)

    @classmethod
    def crate_new_md(cls, name: str, data: str, tag: Tag, parent: Tag or None = None, links: list or None = None):
        if parent is None:
            parent = tag
        if links is None:
            links = []
        return cls(id=get_id(name),
                   name=name,
                   data=data,
                   data_type=DataTypes.MD,
                   tags=[tag],
                   links=links,
                   parent=parent)

    @classmethod
    def crate_new_html(cls, name: str, data: str, tag: Tag, parent: Tag or None = None, links: list or None = None):
        if parent is None:
            parent = tag
        if links is None:
            links = []
        return cls(id=get_id(name),
                   name=name,
                   data=data,
                   data_type=DataTypes.HTML,
                   tags=[tag],
                   links=links,
                   parent=parent)

    @classmethod
    def crate_new_custom(cls, name: str, data: str, tag: Tag, parent: Tag or None = None, links: list or None = None):
        if parent is None:
            parent = tag
        if links is None:
            links = []
        return cls(id=get_id(name),
                   name=name,
                   data=data,
                   data_type=DataTypes.CUSTOM,
                   tags=[tag],
                   links=links,
                   parent=parent)

    def print(self, show=True, debug=False, data=False):
        string_data = f"===========\n{self.data}===========\n --- Nama: {self.name} --- \n --- parent: {self.parent.name} --- "
        if debug:
            string_data += f"\n _________ debug Data _________\n{self.id}\n{self.data_type}\n --- \n"
        i = 0
        for tag in self.tags:
            string_data += f"Tag ({i}) {tag.name} \n"
            string_data += str(tag.related) + "\n" if len(tag.related) != 0 else ""
            string_data += tag.id if debug else ""
            i += 1
        if data:
            string_data = {
                "id": self.id,
                "data": self.data,
                "data_type": self.data_type,
                "tags": self.tags,
                "links": self.links,
                "parent": self.parent
            }
        if show:
            print(string_data)
        else:
            return string_data
