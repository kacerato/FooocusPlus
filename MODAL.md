# FooocusPlus no Modal

Implantação: `modal deploy modal_app.py` a partir deste diretório.

- Aplicação: `fooocusplus-studio`
- Endereço: https://kaceratodev--fooocusplus-studio-studio.modal.run
- GPU: L40S, com alternativas A100 de 40/80 GB quando não houver capacidade; 2 CPUs e 48 GB de RAM. Uma única instância preserva a sessão do Gradio. O contêiner desliga após 10 minutos sem requisições e reinicia no próximo acesso.
- Volume persistente: `fooocusplus-studio-data`, montado em `/data`.
- Login: secret existente `aero-canvas-web-auth` com `AERO_WEB_USER` e `AERO_WEB_PASSWORD`. O servidor falha ao iniciar se qualquer variável estiver ausente.

Modelos podem ser adicionados com `modal run modal_model_download.py::main --version-id ID --file-id ID`. Para arquivos que exigem login, use `--authenticated`, que injeta o Secret `civitai-download` com `CIVITAI_API_KEY`. O script valida o SHA-256 publicado pelo Civitai antes de tornar o arquivo disponível. Os diretórios principais são `/data/models/checkpoints`, `/data/models/loras`, `/data/models/vae` e `/data/models/inpaint`. As imagens e configurações de usuário ficam em `/data/user`, incluindo `/data/user/Outputs`. O FooocusPlus também baixa automaticamente arquivos auxiliares e pesos quando um preset exige.

Checkpoint instalado: `babesIllustriousBy_v55FP16.safetensors`, versão Civitai 2739037, arquivo 2625456, família Illustrious XL. Está disponível no preset `BabesIllustriousV55`. Os pesos Fooocus Inpaint v2.6 estão instalados.

O preset `RealVisXL_Inpaint` oferece o checkpoint `realvisxlV50_v30InpaintBakedvae.safetensors`, Civitai versão 297320, arquivo 234239, base SDXL, versão RealVisXL V3.0 Inpaint (BakedVAE), voltada a fotorrealismo e publicada em janeiro de 2024. A instalação verifica SHA-256 e a entrada de nove canais do UNet. Input Image e Inpaint or Outpaint ficam abertas; Refiner = None, CFG 5, DPM++ 2M SDE Karras, 30 passos, sem estilos adicionais. Instale com `modal run modal_model_download.py::main --version-id 297320 --file-id 234239 --authenticated` e configure com `modal run modal_model_download.py::native_inpaint`.

Checkpoints dedicados de nove canais recebem o latent da imagem mascarada e a máscara no condicionamento positivo e negativo. Eles não recebem o adaptador Fooocus de quatro canais, e exigem imagem/máscara e refiner desativado. A composição final suaviza a borda dentro da máscara, preservando os pixels fora da máscara efetiva (após qualquer erosão/dilatação selecionada). Os downloads CyberRealistic e PornMaster-Anime rejeitados foram arquivados fora da pasta de seleção em `/data/unused-checkpoints`.

Segundo modelo: `Z-Image/pornmasterZImage_turboV35Bf16.safetensors`, versão Civitai 2903129, arquivo 2781142. O preset `pornmasterZImage_turboV35Bf16` usa o workflow ZIT com Qwen_3_4b-Q6_K.gguf e UltraFlux.safetensors. O preset nativo de Z-Image deste fork desativa Input Image; use Illustrious para o inpaint tradicional.

`enhanced/comfy_task.py` foi corrigido para detectar pacotes Z-Image completos pelos componentes no cabeçalho Safetensors. O critério anterior (> 10 GB) classificava incorretamente o BF16 de difusão como pacote completo. A seleção automática de precisão usa o padrão do carregador, preservando os pesos BF16 em vez de forçar FP8.

Este arquivo adapta a implantação ao Modal; `webui.py` exige autenticação do Gradio. O código restante vem do fork FooocusPlus.

## Correção de geração travada

O filtro de arquivos da implantação excluía também `masters/models`, que contém os tokenizers e configurações auxiliares. Além disso, o upstream copia esses arquivos para `UserDir/models`, enquanto a implantação usa `--models-root /data/models`. O servidor agora inclui os arquivos originais e copia apenas os arquivos ausentes para a raiz efetiva antes de iniciar. Isso preserva checkpoints e arquivos personalizados já instalados. Os pesos do Fooocus Expansion continuam sendo baixados pelo fluxo nativo.

O contador de tokens pode usar o tokenizer CLIP real já incluído em `ldm_patched/modules/sd1_tokenizer`. Falhas de inicialização ou execução do worker agora aparecem como erro na geração, em vez de retornar uma galeria vazia ou deixar a fila aguardando indefinidamente. O subprocesso usa saída sem buffer para que os erros apareçam imediatamente nos logs do Modal.

Validação prática em 29/09/2026: sessão autenticada e fila WebSocket do Gradio; geração com Illustrious, 1024×1024, 30 passos; PNG carregado e decodificado por HTTP (1.423.653 bytes). O mesmo PNG foi enviado ao componente de inpaint com máscara retangular; Fooocus Inpaint v2.6 executou os 30 passos e retornou outro PNG 1024×1024, carregado por HTTP (1.438.957 bytes). As duas imagens foram registradas no catálogo persistente. Esta validação confirma geração, envio de imagem, execução de inpaint e entrega dos arquivos; não avalia a qualidade artística nem o segundo modelo Z-Image. A automação do navegador do aplicativo apresentou timeout, então a verificação foi feita pelo protocolo usado pela interface.

## Z-Image Fun Inpaint (ativo)

O estúdio inicia com `ZImage_Fun_Inpaint`. Instalação: `modal run modal_model_download.py::zimage_inpaint`. Todos os pesos são verificados por SHA-256.

- Base: Z-Image Turbo BF16, distribuição Comfy-Org; revisão `6fc90a3b1b653e935a0d175e260736de25b84df5`.
- ControlNet: Alibaba-PAI Fun Union 2.1-2602-8steps; revisão `5155fc56d17821007d6f62ac192c09e0f0e72016`.
- VAE correspondente `z_image_ae.safetensors`; encoder Qwen_3_4b-Q6_K.gguf.
- Fluxo `workflows/ZIT_inpaint_api.json`: imagem de contexto, máscara branca na região editável entregue diretamente ao ZImageFunControlnet (o nó converte internamente para keep-mask), 8 passos Euler/simple, CFG 1, sem expansão de prompt e sem refiner.
- A máscara pintada define a região editável; a área de contexto define o recorte. A borda é suavizada para dentro da máscara na composição final.
- `Preserve Context (Z-Image ControlNet)` controla a força real do ControlNet, recomendação do autor 0,65–1,0; padrão 0,9. O modelo percorre a trajetória completa de 8 passos.
- Os controles de adaptador Fooocus e latent inicial ficam ocultos nesse preset, pois não se aplicam a ele. O preset exige imagem e máscara não vazia, ou seleção de outpaint.

Referência: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1

### Validação Z-Image em 29/09/2026 (horário local)

Sessão autenticada, imagem/máscara enviada ao componente Gradio e geração pela fila WebSocket: 1024×1024, seed 42871, 8 passos, Preserve Context 0,9. Máscara retangular (245,670)–(545,930) sobre um bule vermelho; prompt solicita bule azul. O resultado `/data/user/Outputs/2026-09-30/2026-09-30_02-08-02_9138.png` foi entregue por HTTP 200, decodificado e inspecionado visualmente: bule azul com contexto preservado. Comparação numérica: 78.507 pixels alterados dentro da máscara e zero fora; diferença absoluta média dentro = 44,039 em escala 0–255. Relatório em `/data/user/native-inpaint-validation.json`.

O teste detectou e corrigiu uma inversão duplicada da máscara: `ZImageFunControlnet.diffsynth_controlnet` já faz `1.0 - mask`. O workflow agora envia diretamente a máscara editável. O modelo recriou também a forma do bule dentro da região selecionada; não há garantia de geometria ou identidade idêntica dentro da máscara. A preservação exata fora dela é feita pela composição final. A automação do navegador apresentou timeout; a validação de execução e entrega usou o protocolo real da interface, sem confirmação visual da página no navegador.
