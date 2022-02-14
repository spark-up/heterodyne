#! /bin/bash

if [[ -z "$API_HOST" ]]; then
	exit 1
fi

sudo mysql keystone <<EOF
START transaction;
UPDATE keystone.endpoint
	SET url = regexp_replace(url, 'http://.+:', '$API_HOST:')
	WHERE interface = 'public'
;
SELECT id, url
	FROM keystone.endpoint
	WHERE interface = 'public'
;
COMMIT;
EOF
sudo service memcached restart
