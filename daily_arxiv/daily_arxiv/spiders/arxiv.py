import scrapy
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from semantic_arxiv import load_directions, recall_categories


class ArxivSpider(scrapy.Spider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        directions = load_directions(os.environ.get("DIRECTIONS_CONFIG"))
        if directions:
            categories = recall_categories(directions)
        else:
            categories = os.environ.get("CATEGORIES", "cs.CV").split(",")

        # 保存目标分类列表，用于后续验证
        self.target_categories = set(map(str.strip, categories))
        self.start_urls = [
            f"https://arxiv.org/list/{cat}/new" for cat in self.target_categories
        ]  # 起始URL（计算机科学领域的最新论文）

    name = "arxiv"  # 爬虫名称
    allowed_domains = ["arxiv.org"]  # 允许爬取的域名

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def clean_labeled_text(cls, text, label):
        text = cls.clean_text(text)
        return re.sub(rf"^{re.escape(label)}:\s*", "", text, flags=re.IGNORECASE).strip()

    def parse(self, response):
        # 提取每篇论文的信息
        anchors = []
        for li in response.css("div[id=dlpage] ul li"):
            href = li.css("a::attr(href)").get()
            if href and "item" in href:
                anchors.append(int(href.split("item")[-1]))

        # 遍历每篇论文的详细信息
        for paper in response.css("dl dt"):
            paper_anchor = paper.css("a[name^='item']::attr(name)").get()
            if not paper_anchor:
                continue
                
            paper_id = int(paper_anchor.split("item")[-1])
            if anchors and paper_id >= anchors[-1]:
                continue

            # 获取论文ID
            abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
            if not abstract_link:
                continue
                
            arxiv_id = abstract_link.split("/")[-1]
            
            # 获取对应的论文描述部分 (dd元素)
            paper_dd = paper.xpath("following-sibling::dd[1]")
            if not paper_dd:
                continue
            
            title_text = paper_dd.css(".list-title").xpath("string()").get() or ""
            authors = paper_dd.css(".list-authors a::text").getall()
            comment_text = paper_dd.css(".list-comments").xpath("string()").get() or ""
            subjects_text = paper_dd.css(".list-subjects").xpath("string()").get() or ""
            summary_parts = paper_dd.css("p.mathjax::text").getall()
            summary = self.clean_text(" ".join(summary_parts))
            
            if subjects_text:
                # 解析分类信息，通常格式如 "Computer Vision and Pattern Recognition (cs.CV)"
                # 提取括号中的分类代码
                categories_in_paper = re.findall(r'\(([^)]+)\)', subjects_text)
                
                # 检查论文分类是否与目标分类有交集
                paper_categories = set(categories_in_paper)
                if paper_categories.intersection(self.target_categories):
                    yield {
                        "id": arxiv_id,
                        "categories": list(paper_categories),
                        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
                        "abs": f"https://arxiv.org/abs/{arxiv_id}",
                        "authors": [self.clean_text(author) for author in authors],
                        "title": self.clean_labeled_text(title_text, "Title"),
                        "comment": self.clean_labeled_text(comment_text, "Comments"),
                        "summary": summary,
                    }
                    self.logger.info(f"Found paper {arxiv_id} with categories {paper_categories}")
                else:
                    self.logger.debug(f"Skipped paper {arxiv_id} with categories {paper_categories} (not in target {self.target_categories})")
            else:
                # 如果无法获取分类信息，记录警告但仍然返回论文（保持向后兼容）
                self.logger.warning(f"Could not extract categories for paper {arxiv_id}, including anyway")
                yield {
                    "id": arxiv_id,
                    "categories": [],
                    "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
                    "abs": f"https://arxiv.org/abs/{arxiv_id}",
                    "authors": [self.clean_text(author) for author in authors],
                    "title": self.clean_labeled_text(title_text, "Title"),
                    "comment": self.clean_labeled_text(comment_text, "Comments"),
                    "summary": summary,
                }
