# Guia técnico: reindexação incremental

O índice de conhecimento conserva versões imutáveis dos documentos e calcula um hash do artefato antes da compactação. A reindexação incremental seleciona apenas versões novas.

Durante a recuperação, a política de ACL é aplicada antes do modelo e cada trecho mantém o local de origem. O pacote de contexto é limitado por número de trechos e tokens.
