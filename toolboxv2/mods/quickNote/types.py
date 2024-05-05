import uuid
from dataclasses import dataclass

from toolboxv2.utils.security.cryp import Code


def get_id(name=str(uuid.uuid4())):
    return Code.one_way_hash(name, "quickNote", "id-generator")


@dataclass
class Tag:
    name: str
    id: str
    color: str
    related: list[str]

    @classmethod
    def crate(cls, name: str, related=None, color="#000000"):
        if related is None:
            related = []
        return cls(name=name, id=get_id(name + ":Tag"), related=related, color=color)

    @classmethod
    def crate_root(cls, color="#000000"):
        return cls(name="root", id=get_id("root" + ":Tag"), related=[], color=color)

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


@dataclass
class Note:
    name: str
    id: str
    data: str
    tags: list[str]
    parent: str

    @classmethod
    def crate_root(cls):
        root_tag = Tag.crate_root()
        return cls(id=get_id("root" + ":Note"),
                   name="root",
                   data="",
                   tags=[root_tag.id],
                   parent=root_tag.id)

    @classmethod
    def crate_new_text(cls, name: str, data: str, tags: list[str], parent: str or None = None):
        if len(tags) == 0:
            tags.append(Tag.crate_root().id)
        if parent is None:
            parent = tags[0]
        return cls(id=get_id(name + ":Note"),
                   name=name,
                   data=data,
                   tags=tags,
                   parent=parent)

    def print(self, show=True, debug=False, data=False):
        string_data = f"_________ \n {self.data}\n _________ \n --- Nama: {self.name} --- \n --- parent: {self.parent.name} --- "
        if debug:
            string_data += f"\n =========== debug Data ===========\n{self.id}\n --- \n"
        i = 0
        for tag in self.tags:
            string_data += f"Tag ({i}) {tag} \n"
            i += 1
        if data:
            string_data = {
                "id": self.id,
                "data": self.data,
                "tags": self.tags,
                "parent": self.parent
            }
        if show:
            print(string_data)
        else:
            return string_data
