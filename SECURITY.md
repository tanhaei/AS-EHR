# Security and clinical-data handling

Do not report vulnerabilities by attaching patient data to a public issue.

- Never commit PHI, production credentials, hospital URLs, access tokens, or
  identifiable model prompts/outputs.
- Keep governed data under `data/private/`, which is ignored by Git.
- Treat fine-tuned adapters as potentially memorizing training content until
  privacy testing and institutional review are complete.
- Validate de-identification recall on Persian free text; the demonstration
  regexes in this repository are not certified controls.
- Use an approved private disclosure channel maintained by the project owner
  for security or privacy reports.

