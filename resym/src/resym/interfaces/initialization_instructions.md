# Initialization drafting materials

Read `jobs.json`. Each job includes its prompt, output JSON schema, and the
source identity it describes. Write one JSON response to
`responses/<job_id>.json`. Use precisely that job's response schema; do not
add review status or approval fields. Grounding source belongs in the
`source_code` field as text; do not run candidate source.

`platform.json` contains scanned action signatures, the available query
vocabulary and existing approved contracts. Official CRAM/kRrood interfaces
are trusted. New contracts, native parameter mappings, and generated factory
source still require human review. Do not fabricate missing platform support.
Leave an unimplementable job unanswered and explain its gap to the user.

Responses can be written by an external assistant without API credentials, or
by `resym-init draft`. Both enter the same checks through `resym-init import`.
An import report records submitted, previously submitted, missing and rejected
responses. Import never approves or executes candidates.

Review contracts first on the Viewer's `/capabilities` page. Run `prepare` again
after approval to export the action mapping jobs for those contracts. Review
these mappings on `/capabilities` and factories on `/grounding-factories`.
Keep using the same workspace. Repeated preparation skips artifacts already
pending or approved. Source updates require preparing fresh jobs.
After rejection, prepare again to get a new job with the reviewer's feedback.
