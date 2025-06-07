import redis
import json
from datetime import datetime, timedelta

import redis
import json
from datetime import datetime

# Kết nối đến Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def input_record(msg: str, thread: str = 'Thread-1'):
    record = {
        'time': datetime.now().isoformat(),
        'thread': thread,
        'msg': msg
    }
    # Lưu record vào Redis dưới dạng JSON
    r.rpush('records', json.dumps(record))

def export_records( time: str, thread: 'Thread-1', time_threshold: int = 1000, other_thread_time_threshold: int = 500, msg_threshold: int = 15):
    records = r.lrange('records', 0, -1)
    exported_records = [json.loads(record) for record in records]

    exported_records.reverse()


    # Chuyển đổi thời gian input sang dạng timestamp
    input_time = time.timestamp()

    filtered_records = []
    
    for record in exported_records:
        record_time = datetime.fromisoformat(record['time']).timestamp()
        
        # Kiểm tra điều kiện cho thread trùng và không trùng
        if record['thread'] == thread:
            if record_time < input_time + time_threshold:
                filtered_records.append(record)
        else:
            if record_time < input_time + other_thread_time_threshold:
                filtered_records.append(record)
        
        # Kiểm tra số lượng msg không vượt quá msg_threshold
        if len(filtered_records) >= msg_threshold:
            break

        record_msg = '.'.join(rec['msg'] for rec in filtered_records)

    return record_msg
