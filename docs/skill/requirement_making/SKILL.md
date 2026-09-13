把课程作业说明整理成本工程的 lab 说明文档。

源：`/home/ubuntu/Projects/end_to_end_driving/report/chapterN_homework/requirement/`
目标：`docs/requirement/labN.md`，图片放 `docs/requirement/assets/labN/`

## 原则：以复制为主

`requirement.md` 就是完整正文，**整篇原样复制**，不要改写、精简、重排或补充自己的解读。唯一要改的是图片链接。

包括「作业提交说明」一节也照抄——它是课程原始要求的一部分，留着才有核对依据。本工程更多用于自学和测试，打包提交那套不适用，但这属于**用途说明，写在 README 里**（指向 labN.md 的那句话后面注明「为课程原始要求的完整副本，打包提交说明仅教学场景适用」），不要动 labN.md 正文。同理，README 里用「lab 说明」「lab 报告」「产出物」这类中性说法，labN.md 内部保持原文的「作业」措辞。

## html 不用读

源目录里的 `requirement.html` 是同一份内容的网页版，**内容已完全体现在 md 里，直接忽略**，不要读取、不要复制。

## 大部分文件不用下载

源目录里散落的 `*.jpeg`（形如 `61d72f2...png~tplv-a9rns2rl98-image.jpeg`）是 html 的渲染截图，**md 和 html 都零引用**，内容已在 md 正文里，不要复制。

判断哪些图片真正要处理，只看一处——md 正文里的 `![](...)`：

```bash
grep -oE '!\[[^]]*\]\([^)]+\)' requirement.md
```

拿到清单后**先在源文件夹里找同名文件，找到就直接复制，不要下载**（源仓库的 `assets/` 里往往已经有了）。只有源文件夹确实没有、正文写的是远程链接时才下载。

落地时起个可读文件名（`demo_bev.gif` 而非原始哈希名或带空格的 `Screenshot from ....png`），正文链接写成相对 labN.md 自身的 `assets/labN/xxx.gif`（这样整个 `docs/requirement/` 目录移动时链接不会失效），顺手补上原文缺失的 alt 文本。只处理正文真正引用的图，别把源目录整个搬过来。

## 完成后自检

```bash
diff 源/requirement.md docs/requirement/labN.md   # 差异应仅为图片链接那几行
```

再确认（在 `docs/requirement/` 下执行，图片路径相对 labN.md）每个本地链接都能解析到文件、正文无远程链接残留、图片是有效图片（`file *.gif`，防下到错误页）、`assets/labN/` 里没有未被引用的多余文件。

新增 labN.md 后，记得在 README 的实验安排一节补上指向它的链接。
