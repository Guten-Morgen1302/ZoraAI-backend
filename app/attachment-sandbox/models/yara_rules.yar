/*
  ZoraAI Standard Malware YARA Ruleset
  Generated for Attachment Sandbox
*/

rule Suspicious_Powershell_Download
{
    meta:
        description = "Detects PowerShell download routines commonly used by payload stagers"
        severity = "High"
    strings:
        $ps1 = "powershell" ascii wide nocase
        $ps2 = "Invoke-WebRequest" ascii wide nocase
        $ps3 = "Net.WebClient" ascii wide nocase
        $ps4 = "DownloadString" ascii wide nocase
        $ps5 = "DownloadFile" ascii wide nocase
    condition:
        $ps1 and any of ($ps2,$ps3,$ps4,$ps5)
}

rule Suspicious_Base64_Executable
{
    meta:
        description = "Detects MZ header (PE executable) encoded in Base64 (TVqQ)"
        severity = "Critical"
    strings:
        // TVqQ is the base64 string for "MZ\x90\x00"
        $tvqq1 = "TVqQAAMAAAAEAAAA" ascii wide
        $tvqq2 = "TVqQAAMAAAAEAAA/\\/\\/" ascii wide
    condition:
        any of them
}

rule Embedded_IP_Address_Pattern
{
    meta:
        description = "Detects raw IP addresses embedded in the binary. This is often used by loaders to fetch next-stage payloads or connect to C2s."
        severity = "Medium"
    strings:
        // Generic IP pattern (heuristics only, might trigger FPs but useful with ML)
        $ip_regex = /((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}/
    condition:
        $ip_regex
}

rule Generic_Ransomware_Notes
{
    meta:
        description = "Detects common strings left by generic encrypting ransomware."
        severity = "Critical"
    strings:
        $s1 = "Your files have been encrypted" ascii wide nocase
        $s2 = "restore your files" ascii wide nocase
        $s3 = "Bitcoin" ascii wide nocase
        $s4 = "decrypt" ascii wide nocase
        $s5 = "DECRYPT_FILES" ascii wide nocase
        $s6 = "pay the ransom" ascii wide nocase
    condition:
        3 of them
}

rule Suspicious_API_Usage
{
    meta:
        description = "Detects combinations of APIs commonly used by trojans/packers for process hollowing and memory injection."
        severity = "High"
    strings:
        $api1 = "VirtualAllocEx" ascii nocase
        $api2 = "WriteProcessMemory" ascii nocase
        $api3 = "CreateRemoteThread" ascii nocase
        $api4 = "SetThreadContext" ascii nocase
        $api5 = "NtUnmapViewOfSection" ascii nocase
    condition:
        3 of them
}

rule Suspicious_Macro_AutoStart
{
    meta:
        description = "Detects Office documents containing VBA macros that execute automatically upon opening."
        severity = "High"
    strings:
        $m1 = "AutoOpen" ascii nocase
        $m2 = "Document_Open" ascii nocase
        $m3 = "Workbook_Open" ascii nocase
        $m4 = "AutoExec" ascii nocase
    condition:
        any of them
}

rule EICAR_Test_File
{
    meta:
        description = "Standard antivirus test file."
        severity = "Test"
    strings:
        $eicar = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*" ascii
    condition:
        $eicar
}
