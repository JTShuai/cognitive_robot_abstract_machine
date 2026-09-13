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

`relations` jobs propose relation meanings and typed inputs using the scanned
query vocabulary and supplied native action source. Cite exact query references
and state limitations. Do not supply factory source in this response. An empty
proposal with an explanation is valid when the platform lacks suitable queries.

Importing these replies saves `grounding_requests.json` and proposal evidence in
`grounding_proposals.json`, then appends factory jobs to `jobs.json`. External
authors should read that updated file, write the new replies, and import again.
API drafting continues through the factory jobs in the same invocation. Handwritten
requests are optional. Relation proposals never approve factories or add symbols
to a task library; their meanings and implementations still need human review.

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
