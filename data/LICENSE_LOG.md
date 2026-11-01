# Журнал ліцензій — перші джерела

*За документом «Polite Chainsaw Coding Dataset 200B v0.1», розділ 12. Заповнено Claude з карток датасетів на Hugging Face (30.10.2026). Рішення «включити / відхилити» — за Стефаном.*

| | FineWeb-Edu | FineMath | HPLT 3.0 (українська) | FineWeb-2 (українська, запас) |
|---|---|---|---|---|
| датасет | `HuggingFaceFW/fineweb-edu` | `HuggingFaceTB/finemath` (`finemath-4plus` / `3plus`) | HPLT 3.0, `ukr_Cyrl` (не на HF — з `data.hplt-project.org`) | `HuggingFaceFW/fineweb-2`, `ukr_Cyrl` |
| ревізія | `87f09149ef` (11.07.2025) | `e92b25a616` (06.02.2025) | картка `3394d6ba8d` (08.07.2026), реліз липень 2025 | `af9c13333e` (27.10.2025) |
| ліцензія набору | ODC-By 1.0 | ODC-By 1.0 | CC0 1.0 — **лише на пакування**; тексти авторам не належать HPLT | ODC-By 1.0 |
| додаткові умови | Common Crawl Terms of Use | Common Crawl Terms of Use | Terms of Use HPLT + процедура видалення (takedown) | Common Crawl Terms of Use |
| вимога до назви моделі | немає | немає | немає | немає |
| атрибуція | так (ODC-By: згадати джерело в NOTICE / DATA SOURCES) | так | бажано (цитування) | так |
| персональні дані / секрети | фільтри FineWeb (PII-маскування email/IP) | як FineWeb | є анотація PII, покращений фільтр облікових даних і robots.txt opt-out | є розділ «Personal and Sensitive Information and opt-out» |
| дедуплікація в джерелі | так (MinHash, по дампах) | так | так (глобальна для більшості мов) | так |
| джерело текстів | Common Crawl | Common Crawl | Internet Archive + Common Crawl | Common Crawl |
| перетини | з FineWeb, FineMath, DCLM (спільний Common Crawl) | з FineWeb-Edu, OpenWebMath | з FineWeb-2 uk (обидва з Common Crawl) | з HPLT uk |
| ризики | змішані авторські права вебсторінок (як у всіх CC-корпусів) | те саме | те саме; перевірити машинний переклад і російськомовні вставки (як у твоєму документі) | те саме |
| **рішення** | ✅ включити | ✅ включити | ✅ включити (основне uk) | ✅ включити (запас uk) |

**Мій висновок:** усі чотири — стандартні для відкритого тренування (на них тренують відомі відкриті моделі), без вимог до назви моделі; обов'язок — атрибуція у файлі DATA SOURCES. Для української пропоную HPLT 3.0 як основне (у документі — ~81B токенів у джерелі) і FineWeb-2 як запас, з дедуплікацією між ними.

**Рішення Стефана (30.10.2026):** включаємо все, де не треба перейменовувати модель; атрибуція (згадка джерела) — прийнятна. Ведемо файл DATA SOURCES з атрибуцією.

---

# Повний реєстр джерел документа (перевірено 30.10.2026 з карток Hugging Face)

Правило Стефана: беремо все, де не треба перейменовувати модель; атрибуція — ок.

## ✅ Беремо (дозвільна ліцензія, без вимог до назви)
| джерело | датасет | ліцензія | примітка |
|---|---|---|---|
| **The Stack v3** | `HuggingFaceCode/stack-v3-train` | ODC-By (набір) | вміст файлів усередині (не треба Software Heritage); 3.6T токенів, 713 мов; near-dedup; **ліцензія кожного файлу — окремо** (див. питання нижче); opt-out прибрано |
| OpenCoder FineWeb Code | `OpenCoder-LLM/opc-fineweb-code-corpus` | MIT | вебтексти про код |
| GitHub Agentic PR | `mabujadallah/GitHub-Agentic-PR-Dataset` | CC-BY-4.0 | атрибуція + атрибуція AIDev |
| AgentReviewChat | `Suzhen/AgentReviewChat` | MIT | діалоги code review |
| Themis Git Commits | `project-themis/git-commits` | Apache-2.0 | лише репозиторії під MIT/Apache/BSD/ISC/CC0 |
| OpenWebMath | `open-web-math/open-web-math` | ODC-By | дедуп із FineMath |
| MegaMath | `LLM360/MegaMath` | ODC-By | |
| OpenCoder FineWeb Math | `OpenCoder-LLM/opc-fineweb-math-corpus` | ODC-By | |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | Apache-2.0 | радше для SFT / reasoning |
| FinePDFs-Edu | `HuggingFaceFW/finepdfs-edu` | ODC-By | технічна документація |
| FineWeb | `HuggingFaceFW/fineweb` | ODC-By | загальна англійська |
| DCLM Baseline | `mlfoundations/dclm-baseline-1.0` | CC-BY-4.0 | атрибуція |
| Dolma | `allenai/dolma` | ODC-By | умови компонентів |

## ⚠ Потрібне рішення Стефана
| джерело | датасет | чому |
|---|---|---|
| Proof-Pile-2 | `EleutherAI/proof-pile-2` | «не змінюємо ліцензій складових»: arXiv-статті мають різні ліцензії авторів |
| FineWiki | `HuggingFaceFW/finewiki` | CC-BY-SA-4.0 / GFDL — share-alike |
| Nemotron-CC-Math | `nvidia/Nemotron-CC-Math-v1` | власні умови NVIDIA, доступ після згоди |
| Nemotron-CC-Code | `nvidia/Nemotron-CC-Code-v1` | ручне схвалення доступу, змішані ліцензії (CC-BY / CC-BY-SA / GFDL / other) |
| CodeAlchemy | `open-alchemy/code-alchemy` | IBM: «жодна ліцензія не надається», лише обіцянка не переслідувати |

## ⏭ Не потрібні (замінені The Stack v3)
The Stack v2 Dedup, StarCoderData (закриті умови BigCode), CodeParrot Clean (змішані ліцензії репозиторіїв). Transformers PR — на HF не знайдено.

**Рішення Стефана (30.10.2026) по спірних:** ✅ FineWiki, ✅ Nemotron-CC-Math, ✅ Nemotron-CC-Code (угоду NVIDIA погоджує Стефан сам на HF); ❌ Proof-Pile-2, ❌ CodeAlchemy. **The Stack v3:** дозвільні ліцензії + файли без ліцензії; GPL/AGPL/copyleft — ні.

## Класифікатори якості коду (31.10.2026)
`HuggingFaceTB/stack-edu-classifier-*` (StarEncoder + голова, по мові) — для оцінки якості коду The Stack v3 (метадані-ваги, не тексти). Навчені HuggingFace на оцінках Llama 3.1-70B. **Рішення Стефана:** беремо — наша модель не тренується на виходах Llama і не є похідною від Llama; оцінка-число лише керує відбором даних. Прецедент: SmolLM2 (HF) на Stack-Edu без «Llama» у назві; так само вже взяті FineWeb-Edu / FineMath / FinePDFs-Edu, чиї класифікатори теж на оцінках Llama 3.
