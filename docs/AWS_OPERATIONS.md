# AWS Operations

EC2 백엔드 운영에 필요한 AWS CLI 명령 모음. SSH 차단/IP 변경 같은 자주 발생하는 이슈 위주.

## 환경

- **Region**: `ap-northeast-2` (서울)
- **Instance ID**: `i-0e0cf5289b6ef6125`
- **Public IP**: `13.124.223.49`
- **Security Group**: `sg-005bd1eec8b3c53a3`
- **SSH key**: `~/Documents/AWS/ppt-translator-key.pem`
- **AWS CLI auth**: AWS SSO (`AdministratorAccess` role on account `700388609892`)

## 사전 점검

```bash
aws --version                       # CLI 설치 확인
aws sts get-caller-identity         # 자격증명 유효성 확인
aws configure get region            # 기본 리전 확인
```

자격증명이 만료되었으면 `aws sso login` 으로 재인증.

## 자주 쓰는 작업

### 1. SSH 연결이 타임아웃될 때 (IP 변경)

증상: `ssh: connect to host 13.124.223.49 port 22: Operation timed out`
원인: 보안그룹의 SSH 인바운드 규칙에 등록된 IP가 현재 외부 IP와 다름.

#### 진단

```bash
# 현재 외부 IP 확인
curl -s ifconfig.me

# HTTP(80)와 SSH(22) 도달성 비교 — HTTP만 되면 SG 문제 거의 확정
curl -sf -m 5 http://13.124.223.49/health
nc -zv -w 5 13.124.223.49 22

# 등록된 SSH 인바운드 규칙 확인
aws ec2 describe-security-groups \
  --region ap-northeast-2 \
  --group-ids sg-005bd1eec8b3c53a3 \
  --query 'SecurityGroups[].IpPermissions[?FromPort==`22`]' \
  --output json
```

#### 수정 (기존 IP 교체 = revoke + authorize)

```bash
# 1) 기존 IP 제거
aws ec2 revoke-security-group-ingress \
  --region ap-northeast-2 \
  --group-id sg-005bd1eec8b3c53a3 \
  --protocol tcp --port 22 --cidr <OLD_IP>/32

# 2) 새 IP 추가
aws ec2 authorize-security-group-ingress \
  --region ap-northeast-2 \
  --group-id sg-005bd1eec8b3c53a3 \
  --protocol tcp --port 22 --cidr <NEW_IP>/32
```

> **주의**: `0.0.0.0/0` 또는 광역 CIDR 추가 금지. SSH는 본인 IP(`/32`)만 허용.

### 2. 인스턴스 상태 / 보안그룹 조회

```bash
# IP로 인스턴스 찾기
aws ec2 describe-instances \
  --region ap-northeast-2 \
  --filters "Name=ip-address,Values=13.124.223.49" \
  --query 'Reservations[].Instances[].{ID:InstanceId,State:State.Name,SG:SecurityGroups[].GroupId}' \
  --output json

# 인스턴스 시작/중지
aws ec2 start-instances --region ap-northeast-2 --instance-ids i-0e0cf5289b6ef6125
aws ec2 stop-instances  --region ap-northeast-2 --instance-ids i-0e0cf5289b6ef6125
```

### 3. 배포 흐름 (참고)

`/deploy-ec2` 슬래시 커맨드가 자동화한다. 수동 실행 시:

```bash
ssh -i ~/Documents/AWS/ppt-translator-key.pem ec2-user@13.124.223.49 \
  'cd ~/ppt-translator && git pull origin master && docker compose up -d --build'
```

배포 전 반드시 `git status` 클린 + `git push` 완료 상태여야 한다.

## 트러블슈팅 노트

- **IP가 `.1` ↔ `.2` 처럼 자주 바뀌는 경우**: 동적 IP/CGN 환경. 매번 SG를 갱신하거나, EC2 Instance Connect / SSM Session Manager 도입 고려.
- **HTTP는 되는데 SSH만 안 됨**: 99% 보안그룹 IP 미스매치. 인스턴스 자체는 정상.
- **`aws sts get-caller-identity` 가 토큰 만료 에러**: `aws sso login` 후 재시도.
- **jmespath `||` 우선순위**: `?ToPort==\`22\` || FromPort==\`22\`` 같은 표현식은 의도와 다르게 평가될 수 있음. 단일 조건(`?FromPort==\`22\``)으로 작성 권장.
