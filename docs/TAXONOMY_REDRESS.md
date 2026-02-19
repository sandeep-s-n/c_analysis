# Taxonomy: Redress / LOB View

This taxonomy is designed for **customer redress** at a large bank. It maps CFPB consumer-facing “product” labels to **how redress is organized**: who owns the complaint and is measured on it.

## Why not use CFPB product as LOB?

CFPB categories are consumer-facing (e.g. “Money transfer, virtual currency, or money service”). In a redress function:

- **Money transfer** is usually a *product* or *channel* (wires, ACH, Zelle) under **Consumer Banking** or **Payments**, not its own LOB.
- **Checking or savings** and **Bank account or service** belong to **Consumer Banking** (deposits).
- **Debt or credit management** is often under **Personal Lending** or a dedicated unit, not a standalone LOB.

So we map CFPB product → **redress LOB** (who owns the complaint) and, where useful, → **product_line** (refinement within that LOB).

## Redress LOBs (default)

| LOB | Purpose |
|-----|--------|
| **Consumer Banking** | Deposits, checking, savings, branch services, **payments / money movement** (wires, ACH, Zelle, etc.) |
| **Credit Card** | Card lending, rewards, disputes |
| **Mortgage** | Origination, servicing, foreclosure / loss mitigation |
| **Auto** | Vehicle loan or lease |
| **Personal Lending** | Personal loans, student loans, payday/title if applicable; debt management programs |
| **Debt Collection** | Recovery / collections operations |
| **Credit Reporting & Disputes** | Furnishing, FCRA, dispute handling |
| **Other** | Catch-all |

## Product line (refinement)

Within each LOB, **product_line** allows finer slicing for redress:

- **Consumer Banking** → Deposits | Payments / money movement  
- **Credit Card** → Card lending | Prepaid  
- **Mortgage** → Mortgage  
- **Auto** → Auto  
- **Personal Lending** → Personal loans | Student lending | Personal / title / payday | Debt management  
- **Debt Collection** → Debt collection  
- **Credit Reporting & Disputes** → Credit reporting  

Example: “Money transfer” complaints map to LOB **Consumer Banking** and product_line **Payments / money movement**, so redress can report “Consumer Banking” and still slice “Payments” when needed.

## Customizing for your org

Edit **`data/taxonomy.yaml`**:

1. **`canonical_products`** — Map each CFPB product string to your **redress LOB** name (the list that appears in “Line of business” and emerging topics).
2. **`product_line`** — Map each CFPB product to your **product line** (or sub-LOB) label for drill-down.
3. **`lob_redress`** — Optional: list your LOBs in the order you want them in reports (keys only; values can be empty).

Names and granularity should match how your redress function is structured (e.g. if “Payments” is a separate LOB, add it and map money-transfer products there).
