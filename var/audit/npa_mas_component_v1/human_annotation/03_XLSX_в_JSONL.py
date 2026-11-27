import argparse
import json
from pathlib import Path

from openpyxl import load_workbook

LABEL_KEYS = (
    'exact_reference',
    'lay_paraphrase',
    'multi_issue',
    'applicability_conditions',
    'cross_reference_multi_document',
    'other_unclear',
)
CONFIDENCE = {
    'высокая': 'high',
    'средняя': 'medium',
    'низкая': 'low',
    'high': 'high',
    'medium': 'medium',
    'low': 'low',
}
BASE = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def boolean(value, row_number, column):
    if value is True or value == 1 or str(value).strip().lower() in {'да', 'true'}:
        return True
    if value is False or value == 0 or str(value).strip().lower() in {'нет', 'false'}:
        return False
    if value is None or str(value).strip() == '':
        return None
    raise ValueError(f'Строка {row_number}, столбец {column}: ожидается ДА или НЕТ')


def convert(workbook_path, output_path, kind, allow_draft=False, force=False):
    if output_path.exists() and not force:
        raise ValueError(f'Файл уже существует: {output_path}. Используйте --force только для осознанной замены.')
    source_name = 'npa_query_types_v1_human_b.review_pack.jsonl' if kind == 'secondary' else 'npa_query_types_v1_human_a.review_pack.jsonl'
    expected = read_jsonl(BASE / source_name)
    wb = load_workbook(workbook_path, read_only=False, data_only=False)
    if 'Разметка' not in wb.sheetnames:
        raise ValueError('Отсутствует лист «Разметка»')
    ws = wb['Разметка']
    rows = []
    errors = []
    for row_number in range(2, ws.max_row + 1):
        group_id = ws.cell(row_number, 1).value
        if not group_id:
            continue
        labels = {key: boolean(ws.cell(row_number, column).value, row_number, column) for column, key in zip(range(4, 10), LABEL_KEYS)}
        raw_confidence = str(ws.cell(row_number, 10).value or '').strip().lower()
        confidence = CONFIDENCE.get(raw_confidence)
        reviewed = boolean(ws.cell(row_number, 12).value, row_number, 12) is True
        annotator = str(ws.cell(row_number, 13).value or '').strip()
        row = {
            'query_group': str(group_id),
            'representative_question_id': str(ws.cell(row_number, 2).value or ''),
            'labels': labels,
            'annotation_confidence': confidence,
            'annotator': annotator,
            'notes': str(ws.cell(row_number, 11).value or ''),
            'review_status': 'human_reviewed' if reviewed else 'draft_requires_human_review',
        }
        rows.append(row)
        if not allow_draft:
            if any(value is None for value in labels.values()):
                errors.append(f'{group_id}: заполнены не все шесть признаков')
            if confidence is None:
                errors.append(f'{group_id}: не указана уверенность')
            if not annotator or annotator == 'unassigned':
                errors.append(f'{group_id}: не указан аннотатор')
            if not reviewed:
                errors.append(f'{group_id}: не подтверждена human review')
    expected_ids = [row['query_group'] for row in expected]
    actual_ids = [row['query_group'] for row in rows]
    if actual_ids != expected_ids:
        errors.append('ID групп или их порядок не совпадают с исходным пакетом')
    for row, source in zip(rows, expected):
        if row['representative_question_id'] != source['representative_question_id']:
            errors.append(f"{row['query_group']}: изменён representative_question_id")
    if errors:
        preview = '\n'.join(errors[:30])
        raise ValueError(f'Проверка не пройдена: {len(errors)} ошибок\n{preview}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    temporary.replace(output_path)
    return {
        'valid': True,
        'kind': kind,
        'rows': len(rows),
        'draft': allow_draft,
        'reviewed_rows': sum(row['review_status'] == 'human_reviewed' for row in rows),
        'output': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Конвертация заполненного Excel в JSONL для annotation gate')
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--kind', choices=('primary', 'secondary', 'final'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-draft', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    result = convert(args.workbook.resolve(), args.output.resolve(), args.kind, args.allow_draft, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
