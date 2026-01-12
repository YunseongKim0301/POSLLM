# 코드 입출력 구조
## 입력
총 4종류의 데이터를 입력 받아 활용한다.
1) POS문서 - html 파일 (전체 모드 시 경로: /workspace/pos/phase3/phase3_formatted_new, 소량 모드 시 경로: /workspace/server/uploaded_files)
- html로 구성된 문서로, 여기서 사양값이 존재하는 영역의 chunk를 llm에게 제공해야 한다.
-- table tag로 감싸진 경우 table의 column, row 구조를 이해할 수 있는 형태로 온전히 chunk를 작성한다.
-- 취소선 처리된 데이터는 무시한다
2) 사양값DB - PostgreDB table명: umgv_fin
- header는 다음과 같다.
-- file_name	pmg_lv1	pmg_lv2	pmg_code	pmg_desc	umg_code	umg_desc	extwg	extwg_desc	matnr	doknr	umgv_code	umgv_desc	umgv_value_edit	umgv_uom
-- 일부 컬럼의 값이 전체 또는 일부 비어있을 수 있음
3) 용어집 - PostgreDB table명: pos_dict
- header는 다음과 같다.
-- pmg_code	pmg_desc	umg_code	umg_desc	extwg	extwg_desc	matnr	doknr	umgv_code	umgv_desc	section_num	table_text	value_format	umgv_uom	pos_chunk	pos_extwg_desc	pos_umgv_desc	pos_umgv_value	umgv_value_edit	pos_umgv_uom	evidence_fb
-- 일부 컬럼의 값이 전체 또는 일부 비어있을 수 있음
4) 사양값 template - PostgreDB table명: ext_tmpl
- header는 다음과 같다.
-- pmg_lv1	pmg_lv2	pmg_code	pmg_desc	umg_code	umg_desc	extwg	extwg_desc	matnr	doknr	umgv_code	umgv_desc	umgv_uom
-- 일부 컬럼의 값이 전체 또는 일부 비어있을 수 있음

## 출력
- json 형태로 우선 출력하여 사용자에게 제공한다.
-- json 형태의 각 요소
-- pmg_code	pmg_desc	umg_code	umg_desc	extwg	extwg_desc	matnr	doknr	umgv_code	umgv_desc	section_num	table_text	value_format	umgv_uom	pos_chunk	pos_extwg_desc	pos_umgv_desc	pos_umgv_value	umgv_value_edit	pos_umgv_uom	evidence_fb

## 추출 대상
- 코드는 구조적으로 사양값추출_template.txt 파일을 입력 받아 출력을 내놓는 방식이므로, 다음과 같은 정보(json 구조에는 존재하지만 사양값추출_template.txt 파일에는 없는 column)가 POS 문서에서 필수적으로 추출되어야 한다.
-- section_num	table_text	value_format		pos_chunk	pos_extwg_desc	pos_umgv_desc	pos_umgv_value	umgv_value_edit	pos_umgv_uom	evidence_fb
-- umgv_value_edit는 단위 변환 등 특별한 변경이 불필요한 경우, pos_umgv_value와 동일한 값이다
-- evidence_fb은 우선 빈 값으로 채운다.
