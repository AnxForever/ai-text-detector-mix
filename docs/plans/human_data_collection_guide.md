# Human数据补充采集指南

**创建日期**: 2026-02-09
**目标**: 补充技术类Human数据，解决bert_v6_merged技术文档漏检问题

## 一、问题回顾

当前训练数据严重失衡：
- AI/机器学习内容：92.9% 是 AI 生成
- 技术/编程内容：86.0% 是 AI 生成
- 代码讲解：86.6% 是 AI 生成

**根因**：Human数据主要来自 hc3_human (问答类) 和 thucnews (新闻类)，缺少技术类内容。

## 二、可直接使用的开源数据集

### P0 优先级 - 技术内容

#### 1. CSL 中文科学文献数据集 ⭐ 推荐
- **描述**: 396,209篇中文核心期刊论文元信息（标题、摘要、关键词）
- **来源**: 国家科技资源共享服务工程技术研究中心
- **GitHub**: https://github.com/ydli-ai/CSL
- **HuggingFace**: `ydli-ai/csl`
- **内容**: 2010-2020年期刊论文，13个门类，67个学科
- **项目脚本**: `scripts/data_collection/download_csl.py`

```bash
# 使用项目现有脚本下载
python scripts/data_collection/download_csl.py --count 3000 --output csl_technical
```

#### 2. Chinese-PreTrained-BERT语料
- **描述**: 903万篇百度百科词条
- **GitHub**: https://github.com/SigmaQuan/Awesome-Chinese-Corpus-Datasets-and-Models
- **类型**: 百科知识，包含大量技术词条

#### 3. NKCorpus 高质量中文语料
- **描述**: 700GB高质量中文数据，从Common Crawl清洗而来
- **用途**: 可筛选技术类网页内容
- **参考**: https://blog.csdn.net/weixin_57147647/article/details/128521846

### P1 优先级 - 学术/正式内容

#### 4. CLUE数据集搜索平台
- **GitHub**: https://github.com/CLUEbenchmark/CLUEDatasetSearch
- **包含**: 142个中文NLP数据集
- **推荐**:
  - TNEWS: 今日头条新闻分类
  - THUCNews: 清华新闻语料
  - IFLYTEK: 长文本分类

#### 5. ArXiv中文摘要
- **描述**: 数学417k，物理157万，CS 221k
- **注意**: 需要翻译或筛选中文内容

### P2 优先级 - 评论/社交

#### 6. ChineseNlpCorpus
- **GitHub**: https://github.com/SophonPlus/ChineseNlpCorpus
- **包含**:
  - ChnSentiCorp: 酒店评论
  - waimai_10k: 外卖评论
  - online_shopping: 电商评论
- **项目脚本**: `scripts/data_collection/collect_opensource_human.py`

```bash
# 使用项目现有脚本下载
python scripts/data_collection/collect_opensource_human.py --csl 1000 --toutiao 500 --douban 500
```

## 三、需要爬取的数据源

### 技术博客 (P0)

| 来源 | 网址 | 推荐内容类型 | 爬取难度 |
|-----|-----|------------|---------|
| CSDN | csdn.net | 技术博客、教程 | 中 |
| 掘金 | juejin.cn | 前端/后端/AI | 中 |
| 知乎技术专栏 | zhihu.com | 技术问答、专栏 | 高 |
| 博客园 | cnblogs.com | .NET/Java/Python | 低 |
| SegmentFault | segmentfault.com | 技术问答 | 中 |

**爬取建议**:
- 优先爬取2020年以前的内容（更可能是人类原创）
- 筛选高赞/高阅读量的文章
- 避免爬取AI相关主题（容易混入AI生成内容）

### 技术教程 (P0)

| 来源 | 网址 | 内容类型 |
|-----|-----|---------|
| 菜鸟教程 | runoob.com | 编程入门教程 |
| 廖雪峰教程 | liaoxuefeng.com | Python/Git/JS |
| W3School中文 | w3school.com.cn | Web技术教程 |

### 代码文档 (P0)

| 来源 | 内容类型 | 获取方式 |
|-----|---------|---------|
| GitHub中文README | 项目说明 | GitHub API |
| 官方中文文档 | API文档 | 手动收集 |
| 开源项目Wiki | 技术文档 | GitHub API |

**GitHub中文项目推荐**:
- 筛选 star > 1000 的中文项目
- 收集 README.md 和 docs/ 目录
- 参考: https://github.com/topics/chinese-dataset

### 政府/企业文档 (P1)

| 来源 | 网址 | 内容类型 |
|-----|-----|---------|
| 政府公开信息 | gov.cn | 政策文件、公告 |
| 上市公司公告 | cninfo.com.cn | 年报、招股书 |
| 企业白皮书 | 各公司官网 | 技术白皮书 |

## 四、采集脚本使用

### 现有脚本

```bash
# 1. 下载CSL学术摘要
python scripts/data_collection/download_csl.py --count 3000

# 2. 下载开源评论数据
python scripts/data_collection/collect_opensource_human.py

# 3. 收集正式文本
python scripts/data_collection/collect_formal_human.py
```

### 需要新建的脚本

建议创建以下爬虫脚本:
1. `crawl_csdn_blogs.py` - CSDN技术博客
2. `crawl_github_readme.py` - GitHub中文README
3. `crawl_runoob_tutorials.py` - 菜鸟教程

## 五、数据目标

| 类型 | 目标数量 | 优先级 | 来源 |
|-----|---------|-------|-----|
| 学术论文摘要 | 5,000 | P0 | CSL |
| 技术博客 | 5,000 | P0 | CSDN/掘金/博客园 |
| 技术教程 | 3,000 | P0 | 菜鸟教程/廖雪峰 |
| 代码文档 | 2,000 | P0 | GitHub README |
| 政府公文 | 2,000 | P1 | 政府网站 |
| 企业文档 | 2,000 | P1 | 年报/白皮书 |
| 法律文书 | 1,000 | P2 | 裁判文书网 |

**总计**: ~20,000 条高质量Human技术数据

## 六、快速开始

### 步骤1: 下载现有开源数据

```bash
cd /mnt/c/datacollection

# 下载CSL学术数据 (约3000条)
python scripts/data_collection/download_csl.py --count 3000 --output csl_technical

# 下载评论数据 (约2000条)
python scripts/data_collection/collect_opensource_human.py --csl 1000 --toutiao 500 --douban 500
```

### 步骤2: 检查外部数据集

```bash
# 查看已下载的外部数据集
ls datasets/external/

# 应该包含:
# - M4/
# - DuReader/
# - VCSum/
```

### 步骤3: 整合到训练数据

```bash
# 合并到主训练集
python scripts/data_cleaning/final_merge.py
```

## 七、参考资源

- [CLUE数据集搜索](https://github.com/CLUEbenchmark/CLUEDatasetSearch)
- [CSL科学文献数据集](https://github.com/ydli-ai/CSL)
- [Awesome-Chinese-Corpus](https://github.com/SigmaQuan/Awesome-Chinese-Corpus-Datasets-and-Models)
- [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
- [GitHub中文数据集话题](https://github.com/topics/chinese-dataset)

---

*文档创建时间: 2026-02-09*
