# Хакатон в рамкках конкурса Студент года НИЯУ МИФИ

## Задача: Использование ИИ в продукте

### Описание задачи
При проведении опросов важно не только собрать ответы, но и качественно проанализировать их, чтобы понять реальные мотивы и предпочтения людей.

Представим, что сотрудники отвечают на вопрос: «Что мотивирует вас работать больше?» Ответы могут быть самыми разными: «команда», «коллеги», «зарплата», «бабосики», «шеф», «атмосфера», «амбициозные задачи» и т.д. Сырые данные зачастую избыточны и включают множество синонимов, просторечий или даже нецензурной лексики.

Разработать систему на основе ИИ, которая анализирует список пользовательских ответов возвращает понятное и интерпретируемое облако слов.

Держатель кейса: МТС Линк


### Локальный запуск программы

Инициализация программы:
```
build.bat init
```

Запуск программы осуществляется с помощью следующей команды:
```
python script.py <путь до файл .json>
```

Обновление версии с git до последней на ветке main. ВАЖНО! Удаляет все, что не соответствует новой версии!
```
build.bat update
```