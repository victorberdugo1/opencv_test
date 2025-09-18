//
// ffmpeg -f rawvideo -vcodec rawvideo -s 9152x6944 -pix_fmt gray10le -i capture.raw output.png

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pigpio.h>
#include <termios.h>
#include <fcntl.h>

#define STEP_PIN 27
#define DIR_PIN 17
#define TRIGGER_PIN 22

// Configurar lectura sin bloqueo del teclado
void configurarTerminal()
{
	struct termios term;
	tcgetattr(STDIN_FILENO, &term);
	term.c_lflag &= ~ICANON; // Modo sin buffer (no espera Enter)
	term.c_lflag &= ~ECHO;   // No muestra caracteres en pantalla
	tcsetattr(STDIN_FILENO, TCSANOW, &term);
}

// Restaurar configuración de terminal
void restaurarTerminal()
{
	struct termios term;
	tcgetattr(STDIN_FILENO, &term);
	term.c_lflag |= ICANON | ECHO;
	tcsetattr(STDIN_FILENO, TCSANOW, &term);
}

// Leer una tecla sin bloqueo
int teclaPresionada()
{
	struct timeval tv = {0, 0};
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(STDIN_FILENO, &fds);
	select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
	return FD_ISSET(STDIN_FILENO, &fds);
}

// Función para mover el motor
void move_motor(int step, int direction, int delay)
{
	gpioWrite(DIR_PIN, direction);
	for (int i = 0; i < step; i++)
	{
		gpioWrite(STEP_PIN, 1);
		usleep(delay);
		gpioWrite(STEP_PIN, 0);
		usleep(delay);
	}
}

/*/ Función que ejecuta ffplay para mostrar la preview de la cámara
void tomarPreview()
{
	static int contador = 0;  // Se incrementa cada vez que se llama
	char nombreArchivo[64];
	snprintf(nombreArchivo, sizeof(nombreArchivo), "jpeg_%06d.jpg", contador);

	pid_t pid = fork();
	if (pid == 0)
	{
		execlp("libcamera-jpeg", "libcamera-jpeg",
				"--width", "9152",
				"--height", "6944",
				"--autofocus-mode", "manual",
				"--lens-position", "0",
				"--exposure", "normal",
				"--shutter", "30000",           // en microsegundos
				"--analoggain", "4.0",
				"--denoise", "off",
				"--contrast", "1",
				"--saturation", "1",
				"-t", "1",
				"-o", nombreArchivo,
				NULL);
		perror("execlp (libcamera-jpeg)");
		exit(1);
	}
	else
	{
		waitpid(pid, NULL, 0);
		printf("Imagen JPEG guardada como %s\n", nombreArchivo);
	}

	contador++;
}*/

void tomarPreview()
{
	gpioWrite(TRIGGER_PIN, 1);
	usleep(1000000); // 1.0 sec
	gpioWrite(TRIGGER_PIN, 0);
	usleep(1700000); // 1.7 sec
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Uso: %s <numero_de_pasos>\n", argv[0]);
		return 1;
	}

	if (gpioInitialise() < 0)
	{
		fprintf(stderr, "Error al inicializar GPIO.\n");
		return 1;
	}

	gpioSetMode(STEP_PIN, PI_OUTPUT);
	gpioSetMode(DIR_PIN, PI_OUTPUT);
	gpioSetMode(TRIGGER_PIN, PI_OUTPUT);

	int stepSize = atoi(argv[1]);
	int stepCount = 0;
	int ciclo = 0;

	configurarTerminal();
	printf("Modo: %d pasos por acción\n", stepSize);
	printf("Ejecutando ciclo automático. Presiona 'ESC' para salir o espera que finalicen 2400 ciclos.\n\n");

	while (1)
	{
		if (teclaPresionada())
		{
			char c;
			read(STDIN_FILENO, &c, 1);
			if (c == 27) // Código ASCII de ESC
				break;
		}

		if (ciclo >= 2400)
		{
			printf("Se alcanzaron los 2400 ciclos. Finalizando.\n");
			break;
		}

		stepCount += stepSize;
		printf("Ciclo %d: Moviendo motor %d pasos. Total acumulado: %d pasos\n\n", ciclo + 1, stepSize, stepCount);

		move_motor(stepSize, 0, 1000);
		tomarPreview();

		ciclo++;
	}

	printf("Saliendo...\n");
	gpioWrite(TRIGGER_PIN, 0);
	restaurarTerminal();
	gpioTerminate();
	return 0;
}
