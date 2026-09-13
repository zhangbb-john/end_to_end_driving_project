把课程作业报告整理成本工程的 lab 报告。

源：`/home/ubuntu/Projects/end_to_end_driving/report/chapterN_homework/report.md`
目标：`docs/report/labN_report/report.md`，图片与视频放 `docs/report/labN_report/asset/`

章节与 lab 的对应不是顺序的，按 `report.md` 第一行的 Project 编号核对：lab1=chapter2（Project1）、lab2=chapter3（Project2）、lab3=chapter4（Project3）、lab4=chapter8（Project4）。

## 原则：以复制为主

`report.md` 整篇原样复制，不要改写、精简或补充自己的解读。只改路径——报告里的路径是写给源仓库的，在本工程里全是死链。

## 三类路径都要改

复制后逐一排查，三种写法都出现过：

1. **图片/视频引用**（`src="asset/xxx.png"`、`![](...)`）——把文件复制到 `labN_report/asset/` 下，保持报告里的相对路径可用
2. **markdown 链接**（`[head L1166](../../home_work/Bench2Drive/DriveTransformer/...)`）——源仓库多一层 `home_work/Bench2Drive/` 前缀，本工程里 `DriveTransformer/` 直接在根下；从 `docs/report/labN_report/` 出发是 `../../../DriveTransformer/...`
3. **纯文本路径**（「修改文件」一节列的 `home_work/Bench2Drive/DriveTransformer/...`）——不是链接，容易漏，改成工程内相对路径

```bash
grep -n "home_work" report.md          # 应为空
```

## 附件按引用带

报告正文引用或提到的产出物一并复制到 `asset/`：可视化图片、评测 json、视频。

**视频全部带上**，即使正文没有点名——它们是这个 lab 的实验产出，价值和评测 json 一样。文件名已经能表达含义时（`output_2091_edit_planner.mp4` 说明了 route 和版本）就保持原名，只有像 `output.mp4` 这种含义不明的才改名（`demo_map.mp4`）。

复制进来后要让它们在正文里可达，否则读者无从发现：

- 正文有「（报告外部产出）」这类占位 → 改成指向本地文件的链接
- 正文完全没提 → 在实验结果一节加一个索引表；报告按 route 组织时，用 route 编号对齐视频（见 lab3_report 的做法）
- 章节标题本身就是文件名（`#### 1.5.1. asset/xxx.json`）→ 把文件名包成链接

## 完成后自检

```bash
cd docs/report/labN_report
diff 源/report.md report.md   # 差异应仅为改过的路径行
grep -oE '\]\([^)]+\)' report.md | sed 's/^](//;s/)$//' | grep -v '^http' | while read -r p; do
  f="${p%%#*}"; [ -f "$f" ] && echo "OK $p" || echo "MISS $p"
done
```

再确认 `asset/` 里没有未被引用的多余文件、图片视频是有效文件（`file asset/*`）。

## 与源代码的关系

报告描述的改动应与对应 solution 分支的代码一致。若报告里写了模板中没有的稳健性修正（如 lab1 报告的 `np.ceil`），代码要一并落地，别只留在报告里。
