# Gravador de Áudio do Sistema

Uma aplicação simples para gravar o áudio do sistema usando um cabo virtual e reproduzi-lo posteriormente.

## Pré-requisitos

- Python 3.7+
- VB-Cable instalado

## Instalação

1. Clone este repositório
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Configuração do Cabo Virtual

Para capturar o áudio do sistema (não do microfone), você precisa configurar um cabo de áudio virtual:

1. Baixe e instale o VB-Cable:
   - VB-Cable: https://vb-audio.com/Cable/

2. Configure o cabo virtual como dispositivo de saída padrão do Windows:
   - Clique com o botão direito no ícone de som na bandeja do sistema
   - Selecione "Sons" ou "Configurações de som"
   - Na aba "Reprodução", clique com o botão direito em "CABLE Input" e selecione "Definir como Dispositivo Padrão"

3. O som que você ouve normalmente precisa ser redirecionado para seus fones/alto-falantes:
   - Abra as propriedades do CABLE Output (na aba "Gravação" do painel de som)
   - Vá para a aba "Ouvir"
   - Marque "Ouvir este dispositivo"
   - Selecione seus fones de ouvido/alto-falantes reais em "Reproduzir através deste dispositivo"

## Uso da Aplicação

1. Execute:
   ```
   python src/gui.py
   ```
   Ou:
   ```
   run.bat

2. A aplicação automaticamente detectará seu cabo de áudio virtual
3. Use o botão "Gravar" para iniciar a gravação do áudio do sistema
4. Use o botão "Parar" para finalizar a gravação
5. Use o botão "Reproduzir" para ouvir o áudio gravado

## Solução de Problemas

- Se não conseguir ouvir o áudio do sistema durante a gravação, verifique se configurou corretamente a opção "Ouvir este dispositivo"
- Se a aplicação não gravar o áudio do sistema, verifique se o cabo virtual está instalado corretamente e definido como dispositivo de saída padrão
- Se a aplicação não detectar o cabo virtual, reinicie o computador após a instalação do VB-Cable 