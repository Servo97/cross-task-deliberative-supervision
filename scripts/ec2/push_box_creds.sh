#!/bin/bash
# Push fresh SSO-derived AWS creds to nagababa. Cron-safe: silent no-op when SSO or VPN is down.
# IAM bypass: the BatchOperator SSO role cannot attach an S3 policy to the box's instance role
# (iam:PutRolePolicy denied, 2026-08-01), so the box lives on short-lived creds refreshed from the
# workstation. Install:  crontab -l | { cat; echo "17 */4 * * * bash $HOME/Research/TRI/wsmv2/scripts/ec2/push_box_creds.sh"; } | crontab -
set -uo pipefail
# Cron runs with a bare env: no /usr/local/bin (where aws lives) and no AWS_PROFILE — both caused
# silent "SKIP sso-expired" logs 2026-08-01/02 while interactive pushes worked. Pin them here.
export PATH=/usr/local/bin:/usr/bin:/bin
export AWS_PROFILE="${AWS_PROFILE:-Robotics-LBM-PowerUserAccess-124224456861}"
PEM=/home/sarveshp/Research/TRI/nagababa.pem
BOX=ubuntu@10.242.9.112
LOG=/home/sarveshp/Research/TRI/wsmv2/scripts/ec2/push_box_creds.log
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
if ! ENVOUT=$(aws configure export-credentials --format env-no-export 2>/dev/null); then
  echo "$(date -u +%F' '%T) SKIP sso-expired" >> "$LOG"; exit 0
fi
AKI=$(echo "$ENVOUT" | sed -n 's/^AWS_ACCESS_KEY_ID=//p')
SAK=$(echo "$ENVOUT" | sed -n 's/^AWS_SECRET_ACCESS_KEY=//p')
TOK=$(echo "$ENVOUT" | sed -n 's/^AWS_SESSION_TOKEN=//p')
printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\naws_session_token = %s\n' "$AKI" "$SAK" "$TOK" > "$TMP"
if scp -q -o ConnectTimeout=8 -o BatchMode=yes -i "$PEM" "$TMP" "$BOX":.aws/credentials 2>/dev/null; then
  echo "$(date -u +%F' '%T) OK pushed" >> "$LOG"
else
  echo "$(date -u +%F' '%T) SKIP vpn-or-ssh-down" >> "$LOG"
fi
