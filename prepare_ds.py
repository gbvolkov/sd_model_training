from AIAssistantsLib.assistants.rag_utils.rag_utils import load_vectorstore
from AIAssistantsLib.assistants.rag_assistants import get_retriever, RAGAssistantLocal, RAGAssistantMistralAI
import AIAssistantsLib.config as config

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os

import logging

if __name__ == '__main__':

    model_name = "google/mt5-base"  # or "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    context = """Problem Number: 838
Problem Description: Кей юзеры: Консультанты по процессам
Solution Steps: В данном документе указаны специалисты, к которым следует обращаться за консультацией по процессам. Представлены их имена и часовые пояса.

Additional Information (Part 1/1):
Кей юзеры
Для консультации по процессам вам необходимо обращаться к кей юзерам| _Часовой пояс:_ **_МСК +4_**
Тарасевич Анастасия - Красноярск
Иванова Валентина - Новосибирск
Игнатенко Светлана - Новосибирск
_Часовой пояс:_ **_МСК +2_**
Салимгареева Альбина - Уфа
_Часовой пояс:_ **_МСК_**
Прохорова Мария - Волгоград
---|---

Problem Number: 2926
Problem Description: Инфостарт_2024: User Story Mapping + Impact
Solution Steps: Егор Ткачев проводит мастер-класс по созданию пользовательских историй с использованием метода User Story Mapping и концепции Impact Mapping, помогающих лучше понять потребности пользователей и цели проекта.

Additional Information (Part 1/1):
Как мелкие изменения влекут огромный хаос? ](https://event.infostart.ru/analysis_pm2024/agenda/2047591/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a28/a286a7449ca64108228698f3a539d010.pdf)[Видеозапись](https://www.youtube.com/watch?v=lgD5B66AWyA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665e7fd8d17dd1b8e853f3d)
| 2| Павел Громов| [Мастер-класс: Разработка сбалансированной системы мотивации в проектах: материальное vs нематериальное](https://event.infostart.ru/analysis_pm2024/agenda/2069673/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/f79/f79317d2b679a82b0395b53297c8b74e.pdf)[Видеозапись](https://www.youtube.com/watch?v=j-L-9kkQ_F0)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6666401e8d17dd1b8e87c0ed)
| 2| Максим Логвинов| [Доклад: Тандем проектных команд: Заказчик + Интегратор. Кейс проекта параллельного внедрения 1С:ERP двумя проектными командами. ](https://event.infostart.ru/analysis_pm2024/agenda/2047593/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ea8/ea8aeeec9dc3f6e433eae88b412ad0e6.pdf)[Видеозапись](https://www.youtube.com/watch?v=18V05LeZ9PA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665b0cc8d17dd1b8e8435c1)
| 2| Елизавета Левицкая| [Мастер-класс: Как провести статус-встречу проекта, когда хороших новостей нет или техники аргументации](https://event.infostart.ru/analysis_pm2024/agenda/2053304/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/f16/f16cd9bb0792a516cc5d3669ebd7ca3f.pdf)[Видеозапись](https://www.youtube.com/watch?v=sL0XcJe8QAg)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66674d76ad6bb67350a7084c)
| 2| Ирина Шишкина| [Доклад и мастер-класс: Проект внедрения проектного управления](https://event.infostart.ru/analysis_pm2024/agenda/2047623/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/fa0/fa050c11d9d4c4b1a387f4dcc68af6ae.pdf)[Видеозапись](https://www.youtube.com/watch?v=5S8z-TpKuKQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6667284f53800f9b22ce993d)
| 2| Дмитрий Изыбаев| [Круглый стол: Выгоревший сотрудник: инструкция по применению](https://event.infostart.ru/analysis_pm2024/agenda/2048046/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/087/087feebcc37768777e84585bee8b45b5.pdf)[Видеозапись](https://www.youtube.com/watch?v=KHhKAp7kFa0)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666b2eaeb771e7d817837391&orgid=65cefea79968591b68b7c9ae)
| 2| Анна Бочарова| [Мастер-класс: Разработка наоборот: как распознать продукт внутри и продать его вовне](https://event.infostart.ru/analysis_pm2024/agenda/2067828/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/e8f/e8f14731615d9339d459d2371195874e.pdf)[Видеозапись](https://www.youtube.com/watch?v=17Tt64tGNBQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666abae0b771e7d8177d1888&orgid=65cefea79968591b68b7c9ae)
| 2| Алексей Таченков| [Доклад и мастер-класс: Как быстро описать бизнес-концепцию будущего продукта или нового направления в бизнесе 1С](https://event.infostart.ru/analysis_pm2024/agenda/2047637/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a8d/a8d56264da593200a35583af01630d28.pdf)[Видеозапись](https://www.youtube.com/watch?v=yx0hFZxB61k)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666a4519b771e7d8177856b8&orgid=65cefea79968591b68b7c9ae)
| 3| Александр Прямоносов| [Доклад: Проект, который мог нас "убить": ретроспективный анализ и выводы собственника компании-подрядчика](https://event.infostart.ru/analysis_pm2024/agenda/2053330/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ecb/ecbd95dad88ba56586529cc95be94a5c.pdf)[Видеозапись](https://www.youtube.com/watch?v=P7bQpByJOrk)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66633e514f12255a89352691)
| 3| Сергей Лебедев| [Доклад: Методология и инструменты 1С:PM для поддержки проектной деятельности на разных стадиях развития](https://event.infostart.ru/analysis_pm2024/agenda/2066978/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/d6d/d6d1f6e412ec95389049c83011d95c3b.pdf)[Видеозапись](https://www.youtube.com/watch?v=GTuuFnLfXSs)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666322a04f12255a89339c66)
| 3| Михаил Монтрель| [Доклад: IT проекты и информационная безопасность](https://event.infostart.ru/analysis_pm2024/agenda/2054451/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/22c/22cc9b8262320add446d62ebbedb563f.pdf)[Видеозапись](https://www.youtube.com/watch?v=Hextzs8UUGc)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66621efc4f12255a891e24e2)
| 3| Владимир Ловцов| [Трек по ИИ. Доклад: Ускоряем разработку на 30-60% с помощью ИИ](https://event.infostart.ru/analysis_pm2024/agenda/2050183/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/36d/36d87ac190f04739dcf499d45b0d61ba.pdf)[Видеозапись](https://www.youtube.com/watch?v=4mbY1zC6tCE)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6669e29eb771e7d81776fec4)
| 3| Елена Якубив| [Доклад: Высоконагруженные системы: путь от "реанимации" до "ремиссии"](https://event.infostart.ru/analysis_pm2024/agenda/2070640/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/c62/c6227a73de16a70709247f21848d7da7.pdf)[Видеозапись](https://www.youtube.com/watch?v=HHqG4nBRaoU)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677067ad6bb67350a8ed3a)
| 3| Алексей Таченков| [Форсайт-сессия: Будущее 1С-проектов и продуктов: анализ сценариев](https://event.infostart.ru/analysis_pm2024/agenda/2047612/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/8ce/8ceb5839552d25a8163ec304b5e9f857.pdf)[Видеозапись](https://www.youtube.com/watch?v=NOgjiguwpZs)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666c4fadfc584787a3549192)
| 3| Денис Ермолаев| [Доклад: Нужно ли изобретать велосипед, если хочется на нем прокатиться. Как поставить на поток проекты внедрения ERP](https://event.infostart.ru/analysis_pm2024/agenda/2052369/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/b01/b01c0f74264edd19eefc12b98e020259.pdf)[Видеозапись](https://www.youtube.com/watch?v=glNfo1rpqOA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666c579efc584787a355e855)
Аналитик
**Рекомендация коллег**| **Уровень сложности****(1-простой,****2-средний,****3-сложный)**| **Докладчик(и)**| **Наименование**
---|---|---|---
| 1| Кирилл Анастасин| [Прогностическое мышление для работы и жизни](https://event.infostart.ru/analysis_pm2024/agenda/2036912/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/b73/b73db7e1d503b58c04ff1be047f554cc.pdf)[Видеозапись](https://www.youtube.com/watch?v=soPPipyr6_o)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666227fb4f12255a891e5d9a)
| 1| Дмитрий Изыбаев| [Доклад: Управление качеством входящего сырья, ПФ и ГП](https://event.infostart.ru/analysis_pm2024/agenda/2050209/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/620/6204726063a702882ac402913e46716a.pdf)[Видеозапись](https://www.youtube.com/watch?v=4jtJuQqz6V4)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666447bc8d17dd1b8e750c9e)
| 1| Евгений Горшков| [Доклад: Аналитик. Как быть успешным в профессии. ](https://event.infostart.ru/analysis_pm2024/agenda/2067220/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a94/a94c13deb722331e55e31cf0ca1c3412.pdf)[Видеозапись](https://www.youtube.com/watch?v=nVr_G4A2_tk)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66675f9dad6bb67350a7ead1)
| 1| Алёна Ивахнова| [Трек по ИИ. Практика по тренировке нейросетей](https://event.infostart.ru/analysis_pm2024/agenda/2092132/)[Видеозапись](https://www.youtube.com/watch?v=RuxIdnb7yzM)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666303a34f12255a8930676b)
| 1| Дмитрий Кучма| [Мастер-класс: Работа с претензиями от покупателей и поставщикам](https://event.infostart.ru/analysis_pm2024/agenda/2050207/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/bd2/bd2c93d15dd2ee1853dab61116c7d11d.pdf)[Видеозапись](https://www.youtube.com/watch?v=8sJNDCt343M)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6662253f4f12255a891e478b)
👍️ @Халецкий Станислав Небольшой, но полезный практикум по использованию майнд-мэпов, которые мы не используем в работе, но кот. могут помочь сформировать причинно-следственные связи при работе с крупными задачами и выявить первопричину требуемых изменений в системе.| 1| Анна Щепина| [Практикум: Как аналитику быстро вникнуть в новый проект](https://event.infostart.ru/analysis_pm2024/agenda/2050176/)[Видеозапись](https://www.youtube.com/watch?v=qlmE5Lr0WtQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666450498d17dd1b8e7537ee)
| 1| Егор Ткачев| [Мастер-класс: User Story Mapping + Impact](https://event.infostart.ru/analysis_pm2024/agenda/2070634/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/fc0/fc0aee418e69ae4142d9a7bd238f2f24.pdf)[Видеозапись](https://www.youtube.com/watch?v=8FuBVl2E0rQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66630bfc4f12255a8931c518)
| 1| Алёна Лунина| [Мастер-класс: Обработка больших объемов информации – эффективные инструменты и технологии в работе аналитика](https://event.infostart.ru/analysis_pm2024/agenda/2049242/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a9b/a9bb65338a58df81a88df5cad4da395c.pdf)[Видеозапись](https://www.youtube.com/watch?v=7B1M040OFxY)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665a3ad8d17dd1b8e83d6be)
| 1| Елена Веренич| [Мастер-класс: Бизнес-процессы на примере сквозного учета](https://event.infostart.ru/analysis_pm2024/agenda/2071340/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/606/606158a355ae77ccb7684bfb04107a1f.pdf)[Видеозапись](https://www.youtube.com/watch?v=VqS5tkbIw2g)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677dc7ad6bb67350a9953f)
| 1| Константин Архипов| [Мастер-класс: CJM в продуктовой и in-house разработке](https://event.infostart.ru/analysis_pm2024/agenda/2047636/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/d4a/d4a5aa5c2b54a099e6a8e778a4dae0ee.pdf)[Видеозапись](https://www.youtube.com/watch?v=8XdOUiZf8io)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677cd4ad6bb67350a98559)
| 1| Евгения Александрова| [Доклад: Особенности планирования складского наполнения ТМЦ на ремонтном производстве](https://event.infostart.ru/analysis_pm2024/agenda/2050220/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/1a8/1a85c709b88fa774183e1d35dba5849a.pdf)[Видеозапись](https://www.youtube.com/watch?v=55T7VBQUDNQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665ef028d17dd1b8e85c5c0)
👎🏼 @Лапина Екатерина Мастер-класс посвящен приёмам работы с жалобами. Полезно будет больше для ТП.| 1| Алина Шатрова| [Мастер-класс: Неоценимая ценность жалобы](https://event.infostart.ru/analysis_pm2024/agenda/2048138/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/dd6/dd61f249b335579e96bc6f6712884ec9.pdf)[Видеозапись](https://www.youtube.com/watch?v=ieqW-r1XVq8)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6667788aad6bb67350a94e72)
| 1| Анастасия Лощилова| [Доклад: Естественные противоречия бизнеса, или Как не развалить компанию в процессе автоматизации](https://event.infostart.ru/analysis_pm2024/agenda/2050165/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/0da/0dababd2951acedbfc894bbb68ff2d1e.pdf)[Видеозапись](https://www.youtube.com/watch?v=2CLhP-v-1wI)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666b36ebb771e7d81783b6a3&orgid=65cefea79968591b68b7c9ae)
| 1| Дмитрий Кучма| [Интерактив: Поиграем в процессы](https://event.infostart.ru/analysis_pm2024/agenda/2049228/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/e5e/e5ef9312a18c2f435af9b53bdb6ef3d9.pdf )[Материалы1](https://event.infostart.ru/upload/iblock/_7c996/234/234a3141ced18158ab5c884797bc22f5.pdf)[Видеозапись](https://www.youtube.com/watch?v=DmjtbY4-HwQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666ae568b771e7d81780adbb&orgid=65cefea79968591b68b7c9ae)
| 1| Анастасия Лощилова| [Деловая игра: Меж трех огней: Продажи vs Бухгалтерия vs IT](https://event.infostart.ru/analysis_pm2024/agenda/2050166/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/780/780c50e56790dd2e100d4b7a1eedb756.pdf)[Видеозапись](https://www.youtube.com/watch?v=SxRANRkuOss)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666c50e2fc584787a354c4ee)
| 1| Александр Тихонов
Анатолий Городецкий| [Доклад: От проектирования до практики: как создать идеальный процесс и не сломать его при изменении](https://event.infostart.ru/analysis_pm2024/agenda/2069700/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/f0f/f0ff29c478dd25913d19933b20a86237.pdf)[Видеозапись](https://www.youtube.com/watch?v=qiuLqA8igas)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666b349eb771e7d81783a1f4&orgid=65cefea79968591b68b7c9ae)
| 2| Роман Кальмансон| [Доклад: СППР – технология применения, недостающая функциональность, основные альтернативы – «Архитектура как код» и «Шаблон архитектуры»](https://event.infostart.ru/analysis_pm2024/agenda/2050184/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/957/95763de2a6f4c49544925cecc4925612.pdf)[Видеозапись](https://www.youtube.com/watch?v=78lG06zvJgw)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6669c3f3b771e7d817767c77)
| 2| Станислав Султанов| [Мастер-класс по применению DocHub в 1С, или Создаем единое место правды по архитектуре ваших приложений](https://event.infostart.ru/analysis_pm2024/agenda/2102102/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/d9a/d9a62a230127be31b9c48bc80f1b8748.pdf)[Видеозапись](https://www.youtube.com/watch?v=EP01oog_-Cs)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6662275f4f12255a891e57f7)
👍️👍️👍️ @Яковлев Андрей Интересно для понимания современных возможностей и реальных проблем применения@Грачева Марина @Лапина Екатерина Доклад знакомит с некоторыми решениями, которые существуют сейчас на базе ИИ в 1С - распознавание речи, прогнозирование продаж. Продемонстрированы примеры работы ИИ в 1С: показано что система на данный момент не совершенна, какие ошибки могут возникать при работе с ИИ (не точность оформления текста).| 2| Елена Загибалова| [Трек по ИИ. Доклад: Искусственный интеллект в 1С: инновации и возможности](https://event.infostart.ru/analysis_pm2024/agenda/2051874/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/3f1/3f1cf8c8740b3c331100c4ba2884230d.pdf)[Видеозапись](https://www.youtube.com/watch?v=pFuzocoVzvI)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666431fa8d17dd1b8e747371)
| 2| Татьяна Рыловникова| [Доклад: Как не сойти с ума, закрывая месяц в ERP](https://event.infostart.ru/analysis_pm2024/agenda/2040346/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ba2/ba28b4570a9cf73d3590ca02baac4725.pdf)[Видеозапись](https://www.youtube.com/watch?v=bUw6s-Oq0wM)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6664316a8d17dd1b8e7470d0)
👍️ 👍️ @Грачева Марина @Лапина Екатерина Информативный материал, продемонстрированы схемы интеграций и решения их использования.| 2| Михаил Харитонов| [Доклад: Аналитик VS Интеграции](https://event.infostart.ru/analysis_pm2024/agenda/2069682/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/527/527e7c74bf4045c0186189d6c416b174.pdf)[Видеозапись](https://www.youtube.com/watch?v=OLCAOwayifM)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66631fe04f12255a89336c4d)
👍️ 👍️ @Афанасьева Александра @Шокова Полина Доклад про эффективное управление требованиями. Докладчик рассказывает о видах требований и их смыслах, бизнес ценности, о бережливом подходе в работе аналитика. Также делится подходом к структуре постановки требований, который использует лично в своей практике.| 2| Иннокентий Бодров| [Доклад: Бережливость начинается с бережливого управления требованиями](https://event.infostart.ru/analysis_pm2024/agenda/2051300/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/77d/77d0a8736ba476a0f007bdd2c1f12b31.pdf)[Видеозапись](https://www.youtube.com/watch?v=wrQ77e0112U)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666228fb4f12255a891e6556)
| 2| Евгений Грибков| [Доклад: Консалтинг на проектах внедрения 1С:ERP: польза или вред? ](https://event.infostart.ru/analysis_pm2024/agenda/2049247/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/044/044e6ecbeec250421be99eda51dd0fe1.pdf)[Видеозапись](https://www.youtube.com/watch?v=7gjwRggBouw)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666321424f12255a8933856d)
| 2| Екатерина Милицина| [Доклад: Методики и инструменты бережливого производства и их использование в работе аналитика](https://event.infostart.ru/analysis_pm2024/agenda/2063363/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/db0/db0f1c893ccf9a95a95d669664bc525b.pdf)[Видеозапись](https://www.youtube.com/watch?v=OL9WHu_FBMY)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665f49a8d17dd1b8e863b90)
| 2| Александр Чавалах| [Трек «Отчетность и дашборды». Доклад и мастер-класс: Построение системы управленческой отчетности от идеи до дашбордов. ](https://event.infostart.ru/analysis_pm2024/agenda/2063443/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ab4/ab4c819e90f81a9d7138c1610ceff668.pdf)[Видеозапись](https://www.youtube.com/watch?v=hiBVJQkk69Y)
| 2| Артем Кагукин| [Доклад и мастер-класс: Кибербезопасность. Необходимый минимум](https://event.infostart.ru/analysis_pm2024/agenda/2050193/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/34f/34f9e8852d817e7ed80487bfc4522d32.pdf)[Видеозапись](https://www.youtube.com/watch?v=nKX2zh76wRc)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66644f498d17dd1b8e753417)
| 2| Елена Швец| [Доклад и мастер-класс: Финансовая структура компании: ЦФО, МВЗ, ЦФУ. Ответственность реальная и номинальная – когда система бюджетирования работает, а когда вместо бюджетирования мы получаем лишь цифровую регистрацию будущих и прошедших событий?

Problem Number: 3128
Problem Description: Обучение, конференции, вебинары и т.п.: Кибербезопасность
Solution Steps: Рассматриваются основы кибербезопасности, необходимые меры для обеспечения безопасности информационных систем и предотвращения угроз.

Additional Information (Part 1/1):
Как мелкие изменения влекут огромный хаос? ](https://event.infostart.ru/analysis_pm2024/agenda/2047591/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a28/a286a7449ca64108228698f3a539d010.pdf)[Видеозапись](https://www.youtube.com/watch?v=lgD5B66AWyA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665e7fd8d17dd1b8e853f3d)
| 2| Павел Громов| [Мастер-класс: Разработка сбалансированной системы мотивации в проектах: материальное vs нематериальное](https://event.infostart.ru/analysis_pm2024/agenda/2069673/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/f79/f79317d2b679a82b0395b53297c8b74e.pdf)[Видеозапись](https://www.youtube.com/watch?v=j-L-9kkQ_F0)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6666401e8d17dd1b8e87c0ed)
| 2| Максим Логвинов| [Доклад: Тандем проектных команд: Заказчик + Интегратор. Кейс проекта параллельного внедрения 1С:ERP двумя проектными командами. ](https://event.infostart.ru/analysis_pm2024/agenda/2047593/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ea8/ea8aeeec9dc3f6e433eae88b412ad0e6.pdf)[Видеозапись](https://www.youtube.com/watch?v=18V05LeZ9PA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665b0cc8d17dd1b8e8435c1)
| 2| Елизавета Левицкая| [Мастер-класс: Как провести статус-встречу проекта, когда хороших новостей нет или техники аргументации](https://event.infostart.ru/analysis_pm2024/agenda/2053304/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/f16/f16cd9bb0792a516cc5d3669ebd7ca3f.pdf)[Видеозапись](https://www.youtube.com/watch?v=sL0XcJe8QAg)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66674d76ad6bb67350a7084c)
| 2| Ирина Шишкина| [Доклад и мастер-класс: Проект внедрения проектного управления](https://event.infostart.ru/analysis_pm2024/agenda/2047623/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/fa0/fa050c11d9d4c4b1a387f4dcc68af6ae.pdf)[Видеозапись](https://www.youtube.com/watch?v=5S8z-TpKuKQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6667284f53800f9b22ce993d)
| 2| Дмитрий Изыбаев| [Круглый стол: Выгоревший сотрудник: инструкция по применению](https://event.infostart.ru/analysis_pm2024/agenda/2048046/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/087/087feebcc37768777e84585bee8b45b5.pdf)[Видеозапись](https://www.youtube.com/watch?v=KHhKAp7kFa0)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666b2eaeb771e7d817837391&orgid=65cefea79968591b68b7c9ae)
| 2| Анна Бочарова| [Мастер-класс: Разработка наоборот: как распознать продукт внутри и продать его вовне](https://event.infostart.ru/analysis_pm2024/agenda/2067828/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/e8f/e8f14731615d9339d459d2371195874e.pdf)[Видеозапись](https://www.youtube.com/watch?v=17Tt64tGNBQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666abae0b771e7d8177d1888&orgid=65cefea79968591b68b7c9ae)
| 2| Алексей Таченков| [Доклад и мастер-класс: Как быстро описать бизнес-концепцию будущего продукта или нового направления в бизнесе 1С](https://event.infostart.ru/analysis_pm2024/agenda/2047637/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a8d/a8d56264da593200a35583af01630d28.pdf)[Видеозапись](https://www.youtube.com/watch?v=yx0hFZxB61k)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666a4519b771e7d8177856b8&orgid=65cefea79968591b68b7c9ae)
| 3| Александр Прямоносов| [Доклад: Проект, который мог нас "убить": ретроспективный анализ и выводы собственника компании-подрядчика](https://event.infostart.ru/analysis_pm2024/agenda/2053330/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/ecb/ecbd95dad88ba56586529cc95be94a5c.pdf)[Видеозапись](https://www.youtube.com/watch?v=P7bQpByJOrk)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66633e514f12255a89352691)
| 3| Сергей Лебедев| [Доклад: Методология и инструменты 1С:PM для поддержки проектной деятельности на разных стадиях развития](https://event.infostart.ru/analysis_pm2024/agenda/2066978/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/d6d/d6d1f6e412ec95389049c83011d95c3b.pdf)[Видеозапись](https://www.youtube.com/watch?v=GTuuFnLfXSs)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666322a04f12255a89339c66)
| 3| Михаил Монтрель| [Доклад: IT проекты и информационная безопасность](https://event.infostart.ru/analysis_pm2024/agenda/2054451/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/22c/22cc9b8262320add446d62ebbedb563f.pdf)[Видеозапись](https://www.youtube.com/watch?v=Hextzs8UUGc)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66621efc4f12255a891e24e2)
| 3| Владимир Ловцов| [Трек по ИИ. Доклад: Ускоряем разработку на 30-60% с помощью ИИ](https://event.infostart.ru/analysis_pm2024/agenda/2050183/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/36d/36d87ac190f04739dcf499d45b0d61ba.pdf)[Видеозапись](https://www.youtube.com/watch?v=4mbY1zC6tCE)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6669e29eb771e7d81776fec4)
| 3| Елена Якубив| [Доклад: Высоконагруженные системы: путь от "реанимации" до "ремиссии"](https://event.infostart.ru/analysis_pm2024/agenda/2070640/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/c62/c6227a73de16a70709247f21848d7da7.pdf)[Видеозапись](https://www.youtube.com/watch?v=HHqG4nBRaoU)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677067ad6bb67350a8ed3a)
| 3| Алексей Таченков| [Форсайт-сессия: Будущее 1С-проектов и продуктов: анализ сценариев](https://event.infostart.ru/analysis_pm2024/agenda/2047612/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/8ce/8ceb5839552d25a8163ec304b5e9f857.pdf)[Видеозапись](https://www.youtube.com/watch?v=NOgjiguwpZs)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666c4fadfc584787a3549192)
| 3| Денис Ермолаев| [Доклад: Нужно ли изобретать велосипед, если хочется на нем прокатиться. Как поставить на поток проекты внедрения ERP](https://event.infostart.ru/analysis_pm2024/agenda/2052369/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/b01/b01c0f74264edd19eefc12b98e020259.pdf)[Видеозапись](https://www.youtube.com/watch?v=glNfo1rpqOA)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666c579efc584787a355e855)
Аналитик
**Рекомендация коллег**| **Уровень сложности****(1-простой,****2-средний,****3-сложный)**| **Докладчик(и)**| **Наименование**
---|---|---|---
| 1| Кирилл Анастасин| [Прогностическое мышление для работы и жизни](https://event.infostart.ru/analysis_pm2024/agenda/2036912/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/b73/b73db7e1d503b58c04ff1be047f554cc.pdf)[Видеозапись](https://www.youtube.com/watch?v=soPPipyr6_o)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666227fb4f12255a891e5d9a)
| 1| Дмитрий Изыбаев| [Доклад: Управление качеством входящего сырья, ПФ и ГП](https://event.infostart.ru/analysis_pm2024/agenda/2050209/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/620/6204726063a702882ac402913e46716a.pdf)[Видеозапись](https://www.youtube.com/watch?v=4jtJuQqz6V4)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666447bc8d17dd1b8e750c9e)
| 1| Евгений Горшков| [Доклад: Аналитик. Как быть успешным в профессии. ](https://event.infostart.ru/analysis_pm2024/agenda/2067220/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a94/a94c13deb722331e55e31cf0ca1c3412.pdf)[Видеозапись](https://www.youtube.com/watch?v=nVr_G4A2_tk)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66675f9dad6bb67350a7ead1)
| 1| Алёна Ивахнова| [Трек по ИИ. Практика по тренировке нейросетей](https://event.infostart.ru/analysis_pm2024/agenda/2092132/)[Видеозапись](https://www.youtube.com/watch?v=RuxIdnb7yzM)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666303a34f12255a8930676b)
| 1| Дмитрий Кучма| [Мастер-класс: Работа с претензиями от покупателей и поставщикам](https://event.infostart.ru/analysis_pm2024/agenda/2050207/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/bd2/bd2c93d15dd2ee1853dab61116c7d11d.pdf)[Видеозапись](https://www.youtube.com/watch?v=8sJNDCt343M)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6662253f4f12255a891e478b)
👍️ @Халецкий Станислав Небольшой, но полезный практикум по использованию майнд-мэпов, которые мы не используем в работе, но кот. могут помочь сформировать причинно-следственные связи при работе с крупными задачами и выявить первопричину требуемых изменений в системе.| 1| Анна Щепина| [Практикум: Как аналитику быстро вникнуть в новый проект](https://event.infostart.ru/analysis_pm2024/agenda/2050176/)[Видеозапись](https://www.youtube.com/watch?v=qlmE5Lr0WtQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666450498d17dd1b8e7537ee)
| 1| Егор Ткачев| [Мастер-класс: User Story Mapping + Impact](https://event.infostart.ru/analysis_pm2024/agenda/2070634/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/fc0/fc0aee418e69ae4142d9a7bd238f2f24.pdf)[Видеозапись](https://www.youtube.com/watch?v=8FuBVl2E0rQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66630bfc4f12255a8931c518)
| 1| Алёна Лунина| [Мастер-класс: Обработка больших объемов информации – эффективные инструменты и технологии в работе аналитика](https://event.infostart.ru/analysis_pm2024/agenda/2049242/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/a9b/a9bb65338a58df81a88df5cad4da395c.pdf)[Видеозапись](https://www.youtube.com/watch?v=7B1M040OFxY)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665a3ad8d17dd1b8e83d6be)
| 1| Елена Веренич| [Мастер-класс: Бизнес-процессы на примере сквозного учета](https://event.infostart.ru/analysis_pm2024/agenda/2071340/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/606/606158a355ae77ccb7684bfb04107a1f.pdf)[Видеозапись](https://www.youtube.com/watch?v=VqS5tkbIw2g)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677dc7ad6bb67350a9953f)
| 1| Константин Архипов| [Мастер-класс: CJM в продуктовой и in-house разработке](https://event.infostart.ru/analysis_pm2024/agenda/2047636/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/d4a/d4a5aa5c2b54a099e6a8e778a4dae0ee.pdf)[Видеозапись](https://www.youtube.com/watch?v=8XdOUiZf8io)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=66677cd4ad6bb67350a98559)
| 1| Евгения Александрова| [Доклад: Особенности планирования складского наполнения ТМЦ на ремонтном производстве](https://event.infostart.ru/analysis_pm2024/agenda/2050220/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/1a8/1a85c709b88fa774183e1d35dba5849a.pdf)[Видеозапись](https://www.youtube.com/watch?v=55T7VBQUDNQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6665ef028d17dd1b8e85c5c0)
👎🏼 @Лапина Екатерина Мастер-класс посвящен приёмам работы с жалобами. Полезно будет больше для ТП.| 1| Алина Шатрова| [Мастер-класс: Неоценимая ценность жалобы](https://event.infostart.ru/analysis_pm2024/agenda/2048138/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/dd6/dd61f249b335579e96bc6f6712884ec9.pdf)[Видеозапись](https://www.youtube.com/watch?v=ieqW-r1XVq8)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=6667788aad6bb67350a94e72)
| 1| Анастасия Лощилова| [Доклад: Естественные противоречия бизнеса, или Как не развалить компанию в процессе автоматизации](https://event.infostart.ru/analysis_pm2024/agenda/2050165/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/0da/0dababd2951acedbfc894bbb68ff2d1e.pdf)[Видеозапись](https://www.youtube.com/watch?v=2CLhP-v-1wI)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666b36ebb771e7d81783b6a3&orgid=65cefea79968591b68b7c9ae)
| 1| Дмитрий Кучма| [Интерактив: Поиграем в процессы](https://event.infostart.ru/analysis_pm2024/agenda/2049228/)[Материалы](https://event.infostart.ru/upload/iblock/_7c996/e5e/e5ef9312a18c2f435af9b53bdb6ef3d9.pdf )[Материалы1](https://event.infostart.ru/upload/iblock/_7c996/234/234a3141ced18158ab5c884797bc22f5.pdf)[Видеозапись](https://www.youtube.com/watch?v=DmjtbY4-HwQ)[Аудиозапись и стенограмма](https://api.timelist.ru/transcript?id=666ae568b771e7d81780adbb&orgid=65cefea79968591b68b7c9ae)
| 1| Анастасия Лощи
"""
    input_text = f"Сгенерируй вопрос на основе следующего контекста: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=64)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(question)



    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
        BooleanOptionalAction,
    )
    from langchain_core.output_parsers import StrOutputParser

    vectorestore_path = 'data/vectorstore_e5'

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='query', 
        choices = ['query'],
        help='query - query vectorestore\n'
    )
    args = vars(parser.parse_args())
    mode = args['mode']

    with open('prompts/system_prompt_short.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if mode == 'query':
        assistants = []
        vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
        retriever = get_retriever(vectorestore_path)
        #assistants.append(RAGAssistantGPT(system_prompt, vectorestore_path, output_parser=StrOutputParser))
        #model_name = "mistralai/Ministral-8B-Instruct-2410"
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        #assistants.append(RAGAssistantLocal(system_prompt, vectorestore_path, output_parser=StrOutputParser, model_name=model_name))
        assistants.append(RAGAssistantMistralAI(system_prompt, vectorestore_path, output_parser=StrOutputParser))

        query = ''
        while query != 'stop':
            print('=========================================================================')
            #query = input("Enter your query: ")
            query = "Кто такие key users?"
            if query != 'stop':
                for assistant in assistants:
                    try:
                        reply = assistant.ask_question(query)
                    except Exception as e:
                        logging.error(f'Error: {str(e)}')
                        continue
                    print(f'{type(assistant).__name__} answers:\n{reply['answer']}')
                    print('=========================================================================')
