`uvicorn main:app --reload` - для запуска "_WS_" сервера


`pip install websockets` - необходимо для работы с "_WS_"

# Основной функционал
### Клиентская часть (JavaScript и HTML)
* Начало записи и отправка аудио: При нажатии на кнопку "Старт", запись аудио начинается. Ваш код успешно устанавливает, что каждый доступный фрагмент аудио направляется на сервер сразу после его записи.

* Индикация процесса: Индикаторы интерфейса корректно показывают состояния процесса записи, что важно для пользовательского опыта.

* Остановка записи: При нажатии "Стоп", запись прекращается, и интерфейс возвращается в начальное состояние.

### Серверная часть (Python с FastAPI и WebSocket)
* Прием данных: Сервер корректно принимает байты данных через WebSocket и передает их в модель для распознавания речи.

* Отправка результатов: После получения текста от модели, он отправляется обратно клиенту.

### TODO
* Эффективность модели transcriber:

* Убедитесь, что модель эффективно справляется с потоковыми аудиоданными. Некоторые модели могут требовать специфических форматов входных данных или могут иметь задержки при обработке на лету.
Обработка ошибок:

* Добавьте обработку потенциальных ошибок на сервере при неудачной транскрипции или проблемах с получением данных.
* На стороне клиента также стоит предусмотреть уведомления для пользователя в случае проблем с соединением или на сервере.
Производительность:

* Мониторьте загрузку сервера при работе с непрерывными потоковыми данными, так как распознавание речи — достаточно ресурсоемкий процесс.
