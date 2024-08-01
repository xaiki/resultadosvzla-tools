#!/bin/sh

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0"  \
	--header='User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0' \
	--header='Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/jxl,image/webp,image/png,image/svg+xml,*/*;q=0.8' \
	--header='Accept-Language: en-US,en;q=0.5' \
	--header='Accept-Encoding: gzip, deflate, br, zstd' \
	--header='Alt-Used: resultadosconvzla.com' \
	--header='Connection: keep-alive' \
	--header='Cookie: __cf_bm=P1Pu0nrDbxGCEOoN0DeeS2a0fzizczcwLvnl9BQXg5k-1722494587-1.0.1.1-USIj.nIz5FOxBFRV3aCw66ayneG5nTDx93.bQnkzia06Dko1usB9fS_bQ0WyAqAXIbzy4a.jKTc3jGln_a.E8g' \
	--header='Upgrade-Insecure-Requests: 1' \
	--header='Sec-Fetch-Dest: document' \
	--header='Sec-Fetch-Mode: navigate' \
	--header='Sec-Fetch-Site: cross-site' \
	--header='Sec-GPC: 1' \
	--header='Priority: u=0, i' \
	--header='Pragma: no-cache' \
	--header='Cache-Control: no-cache' \
	$@
