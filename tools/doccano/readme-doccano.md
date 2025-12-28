# Doccano Integration

Complete workflow for invoice field annotation and model training using [Doccano](https://doccano.github.io/doccano/).

## Setup

1. **Install Doccano**:
```bash
pip install doccano
```

2. **Start Doccano server**:
```bash
doccano init
doccano createuser  # Default: admin/password
doccano webserver   # UI at http://localhost:8000/auth
# In another terminal:
doccano task
```

3. **Generate tasks with normalization guards**:
```bash
cd tools/doccano
python tasks_gen.py --seed-folder ../../seed_pdfs --output ./output
```

4. **Import tasks to Doccano**:
- Create new project in Doccano UI (Sequence Labeling)
- Upload `output/tasks.json` to the project
- Use the 10 core invoice fields for labeling

5. **Export and align**:
```bash
# Export from Doccano as JSON
invoicex doccano-import --in path/to/export.json

# Align with pipeline candidates using IoU matching
invoicex doccano-align --all --iou 0.3

# Train XGBoost models
invoicex train
```

## Key Advantages over Label Studio

- **Simpler setup**: Single `pip install doccano` command
- **Lightweight**: Django-based backend, Vue.js frontend  
- **Better API**: RESTful with clean authentication
- **Open source**: MIT license with active development

## Normalization Guard

The system enforces text normalization consistency:
- Each task includes `normalize_version` and `text_checksum`
- Import validates against pipeline's current normalizer  
- Prevents drift when annotation/pipeline versions diverge

## Field Mapping

Doccano labels → Contract fields:
- `InvoiceNumber` → `invoice_number`
- `InvoiceDate` → `invoice_date`
- `TotalAmount` → `total_amount`
- `VendorName` → `vendor_name`
- `CustomerAccount` → `customer_account`

## Line Items

Line item fields are grouped by spatial proximity:
- `LineItemDescription`, `LineItemQuantity`, `LineItemUnitPrice`, `LineItemTotal`
- Ambiguous groupings → `line_items: []` + review queue

## Workflow Summary

```bash
# 1. Generate tasks
python tools/doccano/tasks_gen.py --seed-folder seed_pdfs

# 2. Doccano annotation (manual step in UI)

# 3. Pipeline integration  
invoicex doccano-import --in export.json
invoicex doccano-align --all
invoicex train

# 4. Verify improved accuracy
invoicex run path/to/test.pdf
```
